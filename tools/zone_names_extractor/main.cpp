// ── profiler_names_extractor ───────────────────────────────────────────
// Build-time extraction of ALL profiler-macro string-name literals from
// the kernel + host sources using the libclang C API.
//
// Captured macros:
//   PROFILER_DEVICE_ZONE(ring, "name", lid)  — device zone (hash only)
//   PROFILER_ZONE("name")                     — host zone
//   PROFILER_PLOT("name", val)                — plot value
//   PROFILER_MSG("text") / MSG_FMT("fmt",...) — log messages
//
// Emits a header with a constexpr perfect-hash table:
//
//     namespace profiler {
//     inline constexpr size_t kProfilerNameTableSize = N; // power of 2
//     inline constexpr std::array<std::string_view, N>
//         kProfilerNameTable = {{ ... }};
//     inline constexpr std::string_view lookup_profiler_name(uint32_t hash) {...}
//     }
//
// The tool tries increasing power-of-2 table sizes until it finds one
// with zero collisions (hash % N all unique).  It prints collision stats
// for each size tried so the user can see the load factor.
//
// Usage:
//   profiler_names_extractor --out FILE [--include DIR]... -- SRC...

#include <clang-c/Index.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace {

std::string cxstr(CXString s) {
    std::string r = clang_getCString(s) ? clang_getCString(s) : "";
    clang_disposeString(s);
    return r;
}

/// FNV-1a — must match profiler::device_zone_hash / hash_zone_name.
uint32_t fnv1a(const std::string &s) {
    uint32_t h = 2166136261u;
    for (char c : s) {
        h ^= (uint8_t)c;
        h *= 16777619u;
    }
    return h;
}

/// Profiler macros whose FIRST string-literal argument is a name we want.
static bool is_profiler_name_macro(const std::string &spelling) {
    return spelling == "PROFILER_ZONE" ||
           spelling == "PROFILER_ZONE" ||
           spelling == "PROFILER_ZONE_IN" ||
           spelling == "PROFILER_PLOT" ||
           spelling == "PROFILER_MSG" ||
           spelling == "PROFILER_MSG_FMT" ||
           spelling == "PROFILER_INTEREST_BEGIN";
}

struct CollectData {
    std::set<std::string> *names;
    CXTranslationUnit tu = nullptr;
    unsigned found = 0;
};

/// Extract the first string-literal token from a macro expansion.
static std::string first_string_literal(CXTranslationUnit tu,
                                         CXSourceRange range) {
    CXToken *tokens = nullptr;
    unsigned n = 0;
    clang_tokenize(tu, range, &tokens, &n);
    std::string result;
    for (unsigned i = 0; i < n; ++i) {
        if (clang_getTokenKind(tokens[i]) == CXToken_Literal) {
            std::string tok = cxstr(clang_getTokenSpelling(tu, tokens[i]));
            if (tok.size() >= 2 && tok.front() == '"' && tok.back() == '"')
                result = tok.substr(1, tok.size() - 2);
            break;
        }
    }
    clang_disposeTokens(tu, tokens, n);
    return result;
}

CXChildVisitResult visit_cursor(CXCursor c, CXCursor /*parent*/,
                                 CXClientData d) {
    auto *cd = static_cast<CollectData *>(d);
    if (clang_getCursorKind(c) != CXCursor_MacroExpansion)
        return CXChildVisit_Continue;

    std::string spelling = cxstr(clang_getCursorSpelling(c));
    if (!is_profiler_name_macro(spelling))
        return CXChildVisit_Continue;

    ++cd->found;
    std::string name =
        first_string_literal(cd->tu, clang_getCursorExtent(c));
    if (!name.empty())
        cd->names->insert(name);
    return CXChildVisit_Continue;
}

int extract(const std::vector<std::string> &includes,
            const std::vector<std::string> &sources,
            const std::string &out_path) {
    std::set<std::string> names;
    unsigned expansions = 0;

    for (const auto &src : sources) {
        std::vector<const char *> args = {
            "-x", "c++",
            "-std=c++20",
            // Match the real kernel build: profiler.h only defines the
            // kernel-side API (set_buffer etc.) under KERNEL_BUILD, and
            // KERNEL_NATIVE keeps the TU free of SYCL headers.
            "-DKERNEL_BUILD",
            "-DKERNEL_NATIVE",
            "-Wno-everything",  // tolerate unrelated warnings in headers
        };
        for (const auto &inc : includes) {
            args.push_back("-I");
            args.push_back(inc.c_str());
        }

        CXIndex index = clang_createIndex(/*excludeDeclsFromPCH=*/0,
                                          /*displayDiagnostics=*/0);
        // DetailedPreprocessingRecord is REQUIRED for CXCursor_MacroExpansion
        // cursors to appear in the AST — without it, the visitor sees nothing.
        CXTranslationUnit tu = clang_parseTranslationUnit(
            index, src.c_str(), args.data(), (int)args.size(),
            /*unsaved_files=*/nullptr, /*num_unsaved_files=*/0,
            CXTranslationUnit_DetailedPreprocessingRecord);
        if (!tu) {
            std::fprintf(stderr, "zone_names_extractor: parse failed: %s\n",
                         src.c_str());
            clang_disposeIndex(index);
            continue;
        }

        // Diagnostics: macro-expansion extraction is LEXER-level (the
        // detailed preprocessing record), so it survives semantic errors
        // in the TU — e.g. tonemap_kernel.cpp fails to find
        // sycl/sycl.hpp under -DKERNEL_NATIVE, yet its zones are
        // extracted fine.  Only print diagnostics for TUs that yielded
        // NOTHING — that's the case worth investigating.
        unsigned diag_errors = 0;
        for (unsigned i = 0, nd = clang_getNumDiagnostics(tu); i < nd; ++i) {
            CXDiagnostic diag = clang_getDiagnostic(tu, i);
            if (clang_getDiagnosticSeverity(diag) >= CXDiagnostic_Error)
                ++diag_errors;
            clang_disposeDiagnostic(diag);
        }

        CollectData cd{&names, tu};
        clang_visitChildren(clang_getTranslationUnitCursor(tu),
                            visit_cursor, &cd);
        if (diag_errors && cd.found == 0) {
            std::fprintf(stderr,
                         "profiler_names_extractor: %u errors and no profiler "
                         "expansions in %s\n",
                         diag_errors, src.c_str());
        }
        expansions += cd.found;
        clang_disposeTranslationUnit(tu);
        clang_disposeIndex(index);
    }

    // ── Build (hash, name) vector ──────────────────────────────────
    struct Entry { uint32_t hash; std::string name; };
    std::vector<Entry> entries;
    entries.reserve(names.size());
    for (const auto &n : names)
        entries.push_back({fnv1a(n), n});

    std::printf("profiler_names_extractor: %u expansions, %zu unique names\n",
                expansions, entries.size());

    // ── Hash→index strategies ──────────────────────────────────────
    // We try several index functions; the tool picks the one that
    // yields the smallest collision-free table.
    //
    //   raw      : hash & (N-1)   — low bits only, fast but may cluster
    //   xorfold  : (hash ^ (hash>>16)) & (N-1)  — XOR-fold mixing
    //   prime    : hash % P       — modulo a prime, better distribution
    //
    // For raw and xorfold we try power-of-2 sizes (fast bitwise AND).
    // For prime we try primes >= name_count.

    struct Strategy {
        const char *label;
        std::vector<std::pair<uint32_t, uint32_t>> results; // {size, collisions}
        uint32_t best_size = 0;
    };

    auto try_pow2 = [&](const char *label,
                        auto index_fn) -> Strategy {
        Strategy s{label};
        std::printf("\n  %s (power-of-2):\n", label);
        std::printf("    %8s  %8s  %8s  %s\n",
                    "Size", "Used", "Waste", "Collisions");
        for (uint32_t size = 1u; size <= 16384u; size <<= 1) {
            if (size < entries.size()) {
                std::printf("    %8u  %8zu  %8zu  (too small)\n",
                            size, entries.size(), size - entries.size());
                continue;
            }
            unsigned collisions = 0;
            std::vector<bool> occupied(size, false);
            for (const auto &e : entries) {
                uint32_t slot = index_fn(e.hash) & (size - 1);
                if (occupied[slot]) ++collisions;
                else occupied[slot] = true;
            }
            s.results.push_back({size, collisions});
            std::printf("    %8u  %8zu  %8zu  %8u\n",
                        size, entries.size(), size - entries.size(),
                        collisions);
            if (collisions == 0 && s.best_size == 0)
                s.best_size = size;
        }
        if (s.best_size == 0)
            std::printf("    => no collision-free size up to 16384\n");
        else
            std::printf("    => best: N=%u (%.1f%% used)\n",
                        s.best_size,
                        100.0 * entries.size() / s.best_size);
        return s;
    };

    // Raw low bits
    auto raw = try_pow2("raw", [](uint32_t h) { return h; });

    // XOR-fold: mix high bits into low bits
    uint32_t best_xor_shift = 0;
    auto xorfold = try_pow2("xorfold",
                            [](uint32_t h) { return h ^ (h >> 16); });

    // Also try other XOR shifts to see if one works better
    for (uint32_t shift : {8u, 12u, 13u, 14u, 15u, 16u, 17u, 20u, 24u}) {
        auto s = try_pow2(("xorfold" + std::to_string(shift)).c_str(),
                          [shift](uint32_t h) { return h ^ (h >> shift); });
        if (s.best_size > 0 && s.best_size < xorfold.best_size) {
            xorfold = std::move(s);
            best_xor_shift = shift;
        }
    }

    // Prime modulo: try primes >= name_count
    std::printf("\n  prime (hash %% P):\n");
    std::printf("    %8s  %8s  %8s  %s\n",
                "Prime", "Used", "Waste", "Collisions");
    Strategy prime{"prime"};
    auto is_prime = [](uint32_t n) {
        if (n < 2) return false;
        if (n % 2 == 0) return n == 2;
        for (uint32_t d = 3; d * d <= n; d += 2)
            if (n % d == 0) return false;
        return true;
    };
    for (uint32_t p = (uint32_t)entries.size(); p <= 4096; ++p) {
        if (!is_prime(p)) continue;
        unsigned collisions = 0;
        std::vector<bool> occupied(p, false);
        for (const auto &e : entries) {
            uint32_t slot = e.hash % p;
            if (occupied[slot]) ++collisions;
            else occupied[slot] = true;
        }
        prime.results.push_back({p, collisions});
        std::printf("    %8u  %8zu  %8zu  %8u\n",
                    p, entries.size(), p - entries.size(), collisions);
        if (collisions == 0 && prime.best_size == 0)
            prime.best_size = p;
    }
    if (prime.best_size == 0)
        std::printf("    => no collision-free prime up to 4096\n");
    else
        std::printf("    => best: P=%u (%.1f%% used)\n",
                    prime.best_size,
                    100.0 * entries.size() / prime.best_size);

    // ── Pick the best strategy ────────────────────────────────────
    // We prefer power-of-2 (fast bitwise AND), falling back to prime
    // if it yields a significantly smaller table.
    auto best = &raw;
    if (xorfold.best_size > 0 && xorfold.best_size < best->best_size)
        best = &xorfold;
    if (prime.best_size > 0 && prime.best_size < best->best_size)
        best = &prime;

    if (best->best_size == 0) {
        std::fprintf(stderr,
                     "profiler_names_extractor: FATAL — no collision-free "
                     "table found for %zu names\n",
                     entries.size());
        return 4;
    }

    bool use_prime = (best == &prime);
    bool use_xorfold = (best == &xorfold);
    uint32_t table_size = best->best_size;
    auto index_fn = [&](uint32_t h) -> uint32_t {
        if (use_prime) return h % table_size;
        if (use_xorfold) return (h ^ (h >> best_xor_shift)) & (table_size - 1);
        return h & (table_size - 1);
    };
    bool is_pow2 = !use_prime && ((table_size & (table_size - 1)) == 0);

    std::printf("\n  => selected: %s N=%u (%.1f%% used, %zu wasted)\n",
                best->label, table_size,
                100.0 * entries.size() / table_size,
                table_size - entries.size());

    // ── Emit header ───────────────────────────────────────────────
    std::ofstream f(out_path);
    if (!f) {
        std::fprintf(stderr, "profiler_names_extractor: cannot write %s\n",
                     out_path.c_str());
        return 2;
    }

    // Build the table: fill every slot (empty → "")
    std::vector<std::string> table(table_size);
    for (const auto &e : entries)
        table[index_fn(e.hash)] = e.name;

    f << "// GENERATED FILE — do not edit.\n"
         "// Produced by tools/zone_names_extractor (libclang).\n"
         "// Perfect-hash table: maps every profiler name to a unique slot\n"
         "// with zero collisions.  Strategy: "
      << best->label << ", N=" << table_size << ".\n"
         "#pragma once\n"
         "#include <cstdint>\n"
         "#include <string_view>\n"
         "#include <array>\n"
         "\n"
         "namespace profiler {\n"
         "\n"
         "/// Number of profiler names in the table.\n"
         "inline constexpr size_t kProfilerNameCount = "
      << entries.size() << ";\n"
         "\n"
         "/// Table size"
      << (is_pow2 ? " (power of 2)" : " (prime)") << ".\n"
         "inline constexpr size_t kProfilerNameTableSize = "
      << table_size << ";\n"
         "\n"
         "/// Perfect-hash table.\n"
         "/// Empty slots hold an empty string_view.\n"
         "inline constexpr std::array<std::string_view, "
      << table_size << ">\n"
         "kProfilerNameTable = {{\n";
    for (size_t i = 0; i < table.size(); ++i) {
        if (table[i].empty())
            f << "    /* " << i << " */ \"\",\n";
        else
            f << "    /* " << i << " */ \"" << table[i] << "\",\n";
    }
    f << "}};\n"
         "\n"
         "/// O(1) lookup: hash → name.\n"
         "/// Returns the name if the hash is known, empty string_view "
         "otherwise.\n"
         "inline constexpr std::string_view lookup_profiler_name(uint32_t hash) {\n";

    if (use_xorfold)
        f << "    hash ^= hash >> " << best_xor_shift << ";\n";

    if (is_pow2)
        f << "    auto sv = kProfilerNameTable[hash & (kProfilerNameTableSize - 1)];\n";
    else
        f << "    auto sv = kProfilerNameTable[hash % kProfilerNameTableSize];\n";

    f << "    // Rehash guard: unknown hashes that alias a known slot\n"
         "    // are rejected by recomputing FNV-1a on the stored name.\n"
         "    if (!sv.empty()) {\n"
         "        uint32_t h = 2166136261u;\n"
         "        for (char c : sv) { h ^= (uint8_t)c; h *= 16777619u; }\n"
         "        if (h == hash) return sv;\n"
         "    }\n"
         "    return {};\n"
         "}\n"
         "\n"
         "} // namespace profiler\n";

    std::printf("profiler_names_extractor: wrote %s\n", out_path.c_str());
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    std::vector<std::string> includes, sources;
    std::string out;
    bool after_dash = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (!after_dash && a == "--out" && i + 1 < argc)
            out = argv[++i];
        else if (!after_dash && a == "--include" && i + 1 < argc)
            includes.push_back(argv[++i]);
        else if (!after_dash && a == "--")
            after_dash = true;
        else
            sources.push_back(a);
    }
    if (out.empty() || sources.empty()) {
        std::fprintf(stderr, "usage: zone_names_extractor --out FILE "
                             "[--include DIR]... -- SRC...\n");
        return 1;
    }
    return extract(includes, sources, out);
}
