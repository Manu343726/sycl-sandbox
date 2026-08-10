#pragma once
#include <sycl-sandbox/profiler.h>

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <filesystem>

// Build-time generated perfect-hash table (tools/zone_names_extractor —
// libclang AST extraction of every profiler-macro string literal from
// the kernel + host sources).  Provides O(1) constexpr lookup via
// profiler::lookup_profiler_name(hash).  Generated as a CMake
// custom-command dependency of the app target; when libclang is
// unavailable this is compiled out and the runtime scans below are the
// only source of names.  Must stay OUTSIDE namespace profiler — the
// generated file opens its own profiler namespace.
#ifdef SANDBOX_HAVE_GENERATED_ZONE_NAMES
#include "zone_names.generated.h"
#endif

// ── Profiler name registry (host side) ────────────────────────────────
// Two-tier lookup:
//   1. O(1) constexpr generated table (lookup_profiler_name) — build-time
//      extracted names, zero runtime cost for known hashes.
//   2. Runtime unordered_map — populated by runtime source scanning for
//      names added after the app binary was built (hot-reload).
//
// Thread-safety: the runtime map is guarded by a mutex.  The generated
// table is immutable (constexpr) — no locking needed.  Readers
// (collect_device on the render thread) never block on the generated
// table and only briefly lock for runtime-added names.

namespace profiler {

/// Register a runtime-discovered name (hot-reload fallback).
inline void register_zone_name(uint32_t hash, std::string name) {
    static std::mutex mtx;
    static std::unordered_map<uint32_t, std::string> names;
    std::lock_guard<std::mutex> lock(mtx);
    names[hash] = std::move(name);
}

/// Look up the display name for a hash.
/// Tier 1: O(1) generated table (when available).
/// Tier 2: runtime map (hot-reload fallback).
/// Returns empty string if unknown.
inline std::string lookup_zone_name(uint32_t hash) {
#ifdef SANDBOX_HAVE_GENERATED_ZONE_NAMES
    // Tier 1: constexpr perfect-hash table — O(1), lock-free.
    auto sv = lookup_profiler_name(hash);
    if (!sv.empty()) return std::string(sv);
#endif
    // Tier 2: runtime map (hot-reload names + fallback).
    static std::mutex mtx;
    static std::unordered_map<uint32_t, std::string> names;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = names.find(hash);
    return it != names.end() ? it->second : std::string{};
}

/// All profiler macros whose first string-literal argument is a name
/// we want to extract.  The fallback scanner below matches any of these.
static constexpr const char *kProfilerMacroNames[] = {
    "PROFILER_ZONE",
    "PROFILER_PLOT",
    "PROFILER_MSG",
    "PROFILER_MSG_FMT",
    "PROFILER_MSG_FMT",
};

/// Recursively scan `dir` for profiler-macro string literals and
/// register each name (hash computed with the same FNV-1a as device
/// code).  Only *.h / *.hpp / *.cpp files are read.  Idempotent.
///
/// Used as a fallback when the build-time extractor is unavailable, and
/// as a hot-reload complement: a kernel .so can be rebuilt at runtime
/// with NEW zone/plot/msg names (the app binary — and its compiled-in
/// table — is not rebuilt), so every kernel load re-scans the sources.
inline void scan_zone_names_from_sources(const std::string &dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (dir.empty() || !fs::is_directory(dir, ec)) return;

    for (auto &entry : fs::recursive_directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        auto ext = entry.path().extension();
        if (ext != ".h" && ext != ".hpp" && ext != ".cpp") continue;

        std::ifstream f(entry.path());
        std::string line;
        while (std::getline(f, line)) {
            for (auto macro_name : kProfilerMacroNames) {
                auto pos = line.find(macro_name);
                if (pos == std::string::npos) continue;
                auto q1 = line.find('"', pos);
                if (q1 == std::string::npos) continue;
                auto q2 = line.find('"', q1 + 1);
                if (q2 == std::string::npos) continue;
                std::string name = line.substr(q1 + 1, q2 - q1 - 1);
                if (name.empty()) continue;
                register_zone_name(device_zone_hash(name.c_str()),
                                   std::move(name));
                break;  // one match per line
            }
        }
    }
}

} // namespace profiler
