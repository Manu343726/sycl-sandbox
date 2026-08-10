// `diag mesh` — per-mesh BVH equivalence check.
//
// Builds a synthetic dense triangle mesh (a UV-sphere tessellation, the
// same primitive family the scene loader uses) as the only object, then
// builds TWO scenes from identical geometry:
//   - scene A: per-mesh BVH built (SceneBuilder::build_mesh_bvhs()) —
//     Mesh::hit() traverses the mesh's own BVH
//   - scene B: no per-mesh BVH — Mesh::hit() falls back to the linear
//     scan of its triangle window (the reference)
// and fires N random rays at the mesh handle, verifying both paths
// report identical hits (miss/miss or same t + same u/v).  Also checks
// every built mesh BVH is structurally valid: roots in range, root
// bounds == the mesh's triangle-union AABB, leaves reference triangles
// inside the mesh's window.

#include "diags.h"
#include "mesh_diag.h"

#include <sycl-sandbox/rt/aabb.h>
#include <sycl-sandbox/rt/types.h>
#include <sycl-sandbox/scene/data.h>
#include <sycl-sandbox/rt/hittables/triangle.h>
#include <sycl-sandbox/rt/materials/lambertian.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace rt;

namespace {

bool check(bool cond, const char *what) {
    std::printf("%s: %s\n", cond ? "PASS" : "FAIL", what);
    return cond;
}

/// Tessellate a UV sphere into `stacks` × `slices` triangles
/// (2 triangles per quad) — a dense, non-trivial mesh for the BVH.
std::vector<hittables::Triangle> make_sphere_mesh(int stacks, int slices,
                                                  float radius,
                                                  float3 center) {
    std::vector<hittables::Triangle> tris;
    tris.reserve(stacks * slices * 2);
    auto V = [&](int i, int j) -> float3 {
        float u = (float)j / (float)slices;
        float v = (float)i / (float)stacks;
        float theta = v * 3.14159265f;
        float phi = u * 2.f * 3.14159265f;
        float x = radius * std::sin(theta) * std::cos(phi);
        float y = radius * std::cos(theta);
        float z = radius * std::sin(theta) * std::sin(phi);
        return {center.x + x, center.y + y, center.z + z};
    };
    for ( int i = 0; i < stacks; i++ ) {
        for ( int j = 0; j < slices; j++ ) {
            float3 a = V(i, j), b = V(i + 1, j);
            float3 c = V(i + 1, j + 1), d = V(i, j + 1);
            tris.push_back(hittables::Triangle {a, b, c});
            tris.push_back(hittables::Triangle {a, c, d});
        }
    }
    return tris;
}

/// Structurally validate one mesh's BVH stored in `scene`.
/// Checks: root in range, root bounds == the triangle-union AABB, and
/// every leaf's triangle index lies inside the mesh's window.
bool validate_mesh_bvh(const SceneView &scene, uint32_t mesh_index) {
    const auto &m = scene.meshes[mesh_index];
    if ( m.bvh_root < 0 ) {
        std::printf("  mesh[%u]: no per-mesh BVH (bvh_root=-1)\n", mesh_index);
        return true; // linear-scan fallback is legal
    }
    if ( m.bvh_root >= scene.num_mesh_bvh_nodes ) {
        std::printf("  mesh[%u]: bvh_root %d out of range (%d nodes)\n",
                    mesh_index, m.bvh_root, scene.num_mesh_bvh_nodes);
        return false;
    }

    // Union AABB of the mesh's triangles (the reference).
    Aabb union_box = scene.triangles[m.first_triangle].aabb();
    for ( uint32_t i = 1; i < m.num_triangles; i++ ) {
        union_box = aabb_merge(union_box,
                               scene.triangles[m.first_triangle + i].aabb());
    }

    // Walk the flat array from the root and verify every node.
    const BvhNode &root = scene.mesh_bvh_nodes[m.bvh_root];
    bool ok = true;
    if ( std::fabs(root.bounds.min.x - union_box.min.x) > 1e-5f ||
         std::fabs(root.bounds.min.y - union_box.min.y) > 1e-5f ||
         std::fabs(root.bounds.min.z - union_box.min.z) > 1e-5f ||
         std::fabs(root.bounds.max.x - union_box.max.x) > 1e-5f ||
         std::fabs(root.bounds.max.y - union_box.max.y) > 1e-5f ||
         std::fabs(root.bounds.max.z - union_box.max.z) > 1e-5f ) {
        std::printf("  mesh[%u]: root bounds != triangle-union AABB\n",
                    mesh_index);
        ok = false;
    }
    // The mesh's BVH nodes are contiguous starting at bvh_root (the
    // builder appends one whole tree per mesh).  A pre-order tree of
    // n leaves has exactly 2n-1 nodes.
    int tree_nodes = 2 * (int)m.num_triangles - 1;
    if ( m.bvh_root + tree_nodes > scene.num_mesh_bvh_nodes ) {
        std::printf("  mesh[%u]: BVH extends past the node array\n",
                    mesh_index);
        ok = false;
    }
    // Verify all leaves reference triangles inside the mesh's window, and
    // every internal child link stays inside this tree's own node range
    // [bvh_root, bvh_root + 2n-1).  The latter catches trees concatenated
    // into the shared array with un-offset (local, 0-based) child links.
    uint32_t stack[64];
    int stack_size = 0;
    stack[stack_size++] = (uint32_t)m.bvh_root;
    while ( stack_size > 0 ) {
        uint32_t node_index = stack[--stack_size];
        if ( node_index < (uint32_t)m.bvh_root ||
             node_index >= (uint32_t)m.bvh_root + (uint32_t)tree_nodes ) {
            std::printf("  mesh[%u]: node %u outside tree range [%d,%d)\n",
                        mesh_index, node_index, m.bvh_root,
                        m.bvh_root + tree_nodes);
            ok = false;
            continue;
        }
        const BvhNode &node = scene.mesh_bvh_nodes[node_index];
        if ( node.left == BVH_LEAF ) {
            if ( node.right < m.first_triangle ||
                 node.right >= m.first_triangle + m.num_triangles ) {
                std::printf("  mesh[%u]: leaf references triangle %u "
                            "outside window [%u,%u)\n",
                            mesh_index, node.right, m.first_triangle,
                            m.first_triangle + m.num_triangles);
                ok = false;
            }
        } else {
            stack[stack_size++] = node.left;
            stack[stack_size++] = node.right;
        }
    }
    return ok;
}

int run_mesh_diag(int num_rays) {
    rt::Runtime rt; // null queue -> software mode

    auto mesh_tris = make_sphere_mesh(48, 96, 1.f, {0, 0, 0});
    int mesh_tri_count = (int)mesh_tris.size();

    // Scene A: per-mesh BVH built.
    SceneBuilder b;
    b.add_mesh(mesh_tris, materials::lambertian({0.8f, 0.2f, 0.2f}));
    b.build_bvh();
    b.build_mesh_bvhs();
    SceneData data_bvh = b.build(&rt);

    // Scene B: identical geometry, NO per-mesh BVH (linear-scan reference).
    SceneBuilder b2;
    b2.add_mesh(mesh_tris, materials::lambertian({0.8f, 0.2f, 0.2f}));
    b2.build_bvh();
    SceneData data_lin = b2.build(&rt);

    SceneView v_bvh = data_bvh.view();
    SceneView v_lin = data_lin.view();
    const auto &m_bvh = v_bvh.meshes[0];
    const auto &m_lin = v_lin.meshes[0];

    std::printf("mesh: tris=%d (stacks=48 slices=96) bvh_root=%d "
                "mesh_bvh_nodes=%d (expect %d)\n",
                mesh_tri_count, m_bvh.bvh_root, v_bvh.num_mesh_bvh_nodes,
                2 * (int)m_bvh.num_triangles - 1);
    std::printf("linear fallback: bvh_root=%d\n", m_lin.bvh_root);

    bool ok = validate_mesh_bvh(v_bvh, 0);

    // Equivalence: per-mesh-BVH hits == linear-scan hits on random rays.
    std::mt19937 gen(9876);
    std::uniform_real_distribution<float> ud(-3.f, 3.f);
    int mismatches = 0, hits = 0, misses = 0;
    for ( int i = 0; i < num_rays; i++ ) {
        Ray ray;
        ray.orig = {ud(gen), ud(gen), ud(gen)};
        ray.dir = {ud(gen), ud(gen), ud(gen)};
        float len = std::sqrt(ray.dir.x * ray.dir.x + ray.dir.y * ray.dir.y +
                              ray.dir.z * ray.dir.z);
        if ( len < 1e-6f ) { --i; continue; }
        ray.dir = {ray.dir.x / len, ray.dir.y / len, ray.dir.z / len};

        auto h_bvh = m_bvh.hit(ray, 0.001f, 1e30f, v_bvh.triangles,
                               v_bvh.mesh_bvh_nodes);
        auto h_lin = m_lin.hit(ray, 0.001f, 1e30f, v_lin.triangles,
                               nullptr);

        if ( !h_bvh ) {
            if ( h_lin ) { ++mismatches; }
            else { ++misses; }
            continue;
        }
        if ( !h_lin ) { ++mismatches; continue; }
        ++hits;
        float dt = std::fabs(h_bvh->t - h_lin->t);
        float du = std::fabs(h_bvh->u - h_lin->u);
        float dv = std::fabs(h_bvh->v - h_lin->v);
        if ( dt > 1e-4f || du > 1e-4f || dv > 1e-4f ) {
            ++mismatches;
            if ( mismatches <= 5 ) {
                std::printf("MISMATCH t_bvh=%f t_lin=%f u=%f/%f v=%f/%f\n",
                            h_bvh->t, h_lin->t, h_bvh->u, h_lin->u,
                            h_bvh->v, h_lin->v);
            }
        }
    }
    std::printf("rays=%d hits=%d misses=%d mismatches=%d\n",
                num_rays, hits, misses, mismatches);
    bool equiv = mismatches == 0;
    if ( equiv ) {
        std::printf("PASS: mesh-BVH hit == linear scan on %d rays\n",
                    num_rays);
    }

    // ── Trace-collector integration ───────────────────────────────────
    // An armed collector must record per-mesh BVH node events while the
    // per-mesh BVH path traverses, and record none on the linear fallback
    // — the scene-debug view highlights exactly those events.
    {
        constexpr uint32_t STEP_CAP = 1u << 4;
        constexpr uint32_t EVENT_CAP = 1u << 10;
        TraceCollectorHeader h_bvh = {};
        TraceStepRecord steps_bvh[STEP_CAP];
        TraceEvent evs_bvh[EVENT_CAP];
        Context ctx_bvh;
        ctx_bvh.collector = TraceCollector {&h_bvh, evs_bvh, steps_bvh,
                                            EVENT_CAP, STEP_CAP};
        Ray down{{0.f, 2.5f, 0.f}, {0.f, -1.f, 0.f}, 0.f};
        auto hv = m_bvh.hit(down, 1e-3f, 1e30f, v_bvh.triangles,
                            v_bvh.mesh_bvh_nodes, ctx_bvh);
        int bvh_events = 0;
        for ( uint32_t i = 0; i < h_bvh.event_pos; i++ ) {
            if ( evs_bvh[i & (EVENT_CAP - 1)].kind ==
                 (uint8_t)TraceEventKind::HittableBvhNode ) {
                bvh_events++;
            }
        }
        ok &= check(hv && bvh_events > 0,
                    "collector: per-mesh BVH node events recorded");

        TraceCollectorHeader h_lin = {};
        TraceStepRecord steps_lin[STEP_CAP];
        TraceEvent evs_lin[EVENT_CAP];
        Context ctx_lin;
        ctx_lin.collector = TraceCollector {&h_lin, evs_lin, steps_lin,
                                            EVENT_CAP, STEP_CAP};
        m_lin.hit(down, 1e-3f, 1e30f, v_lin.triangles, nullptr, ctx_lin);
        int lin_events = 0;
        for ( uint32_t i = 0; i < h_lin.event_pos; i++ ) {
            if ( evs_lin[i & (EVENT_CAP - 1)].kind ==
                 (uint8_t)TraceEventKind::HittableBvhNode ) {
                lin_events++;
            }
        }
        ok &= check(lin_events == 0,
                    "collector: no per-mesh BVH events on linear fallback");
        std::printf("collector: bvh-path events=%d, linear-path events=%d\n",
                    bvh_events, lin_events);
    }

    // ── Culling check: BVH must NOT scan the whole window ─────────────
    // The point of the per-mesh BVH is to skip triangles.  Fire a narrow
    // ray (straight down onto the pole) and count how many Triangle
    // hit-test events the collector sees on the BVH path vs the linear
    // fallback.  A real traversal touches only a handful of triangles
    // near the entry point; a linear scan touches every triangle in the
    // window (the user-reported "computes all triangles then checks the
    // BVH" symptom).  Requires that < 20% of the window is actually
    // tested.
    {
        constexpr uint32_t STEP_CAP = 1u << 4;
        constexpr uint32_t EVENT_CAP = 1u << 14;
        auto count_tri_tests = [&](const SceneView &v, const BvhNode *bvh) {
            TraceCollectorHeader h = {};
            TraceStepRecord steps[STEP_CAP];
            TraceEvent evs[EVENT_CAP];
            Context ctx;
            ctx.collector = TraceCollector {&h, evs, steps, EVENT_CAP,
                                            STEP_CAP};
            Ray down{{0.f, 2.5f, 0.f}, {0.f, -1.f, 0.f}, 0.f};
            v.meshes[0].hit(down, 1e-3f, 1e30f, v.triangles, bvh, ctx);
            int tri_tests = 0, bvh_nodes = 0;
            for ( uint32_t i = 0; i < h.event_pos; i++ ) {
                auto &e = evs[i & (EVENT_CAP - 1)];
                if ( e.kind == (uint8_t)TraceEventKind::HitTest &&
                     e.payload == (uint8_t)HittableType::Triangle ) {
                    tri_tests++;
                }
                if ( e.kind == (uint8_t)TraceEventKind::HittableBvhNode ) {
                    bvh_nodes++;
                }
            }
            return std::make_pair(tri_tests, bvh_nodes);
        };
        auto c_bvh = count_tri_tests(v_bvh, v_bvh.mesh_bvh_nodes);
        auto c_lin = count_tri_tests(v_lin, nullptr);
        // The builder drops degenerate (zero-area) triangles when adding
        // a mesh, so the window is smaller than the raw tessellation count
        // (192 pole triangles are dropped here).
        int window = (int)v_lin.meshes[0].num_triangles;
        std::printf("culling: window=%d tris, BVH tests=%d (%.2f%%) via %d "
                    "nodes, linear tests=%d (100%%)\n",
                    window, c_bvh.first, 100.0 * c_bvh.first / window,
                    c_bvh.second, c_lin.first);
        bool culls = c_bvh.first < window / 5 &&
                     c_lin.first == window;
        ok &= check(culls,
                    "BVH tests a fraction of the window (not a linear scan)");
        ok &= check(c_bvh.first < c_lin.first,
                    "BVH tests fewer triangles than the linear fallback");
    }

    data_bvh.free(&rt);
    data_lin.free(&rt);
    ok = ok && equiv;
    if ( !equiv ) {
        std::printf("FAIL: single-mesh equivalence\n");
    }

    // ── Multi-mesh scene: concatenation guard ──────────────────────────
    // Per-mesh trees share one flat node array, so every mesh but the
    // first lives at a NON-ZERO offset.  If the builder forgets to offset
    // the trees' local child links, only the first mesh traverses and the
    // rest silently miss.  Build a 3-mesh scene and verify each mesh's
    // BVH path matches its linear-scan reference.
    {
        auto m1 = make_sphere_mesh(24, 48, 0.6f, {-2.f, 0.6f, 0.f});
        auto m2 = make_sphere_mesh(12, 24, 0.3f, {1.5f, 0.3f, 1.f});
        // A tiny 4-triangle tetrahedron (fewer than a BVH node's worth
        // of leaves — stresses small trees).
        std::vector<hittables::Triangle> m3;
        m3.push_back({{0, 0, 0}, {1, 0, 0}, {0.5f, 0, 0.8f}});
        m3.push_back({{0, 0, 0}, {1, 0, 0}, {0.5f, 1.f, 0.4f}});
        m3.push_back({{0, 0, 0}, {0.5f, 0, 0.8f}, {0.5f, 1.f, 0.4f}});
        m3.push_back({{1, 0, 0}, {0.5f, 1.f, 0.4f}, {0.5f, 0, 0.8f}});

        SceneBuilder bA;
        bA.add_mesh(m1, materials::lambertian({1.f, 0.f, 0.f}));
        bA.add_mesh(m2, materials::lambertian({0.f, 1.f, 0.f}));
        bA.add_mesh(m3, materials::lambertian({0.f, 0.f, 1.f}));
        bA.build_bvh();
        bA.build_mesh_bvhs();
        SceneData dA = bA.build(&rt);
        SceneData dB = [&] {
            SceneBuilder bB;
            bB.add_mesh(m1, materials::lambertian({1.f, 0.f, 0.f}));
            bB.add_mesh(m2, materials::lambertian({0.f, 1.f, 0.f}));
            bB.add_mesh(m3, materials::lambertian({0.f, 0.f, 1.f}));
            bB.build_bvh();
            return bB.build(&rt); // no per-mesh BVH -> linear reference
        }();
        SceneView vA = dA.view(), vB = dB.view();

        bool multi_ok = true;
        for ( uint32_t m = 0; m < vA.num_meshes; m++ ) {
            multi_ok &= validate_mesh_bvh(vA, m);
            // Union AABB of this mesh's triangles, cast a ray through it.
            Aabb box = vA.triangles[vA.meshes[m].first_triangle].aabb();
            for ( uint32_t i = 1; i < vA.meshes[m].num_triangles; i++ ) {
                box = aabb_merge(box,
                    vA.triangles[vA.meshes[m].first_triangle + i].aabb());
            }
            float3 c = {(box.min.x + box.max.x) * 0.5f,
                        (box.min.y + box.max.y) * 0.5f,
                        (box.min.z + box.max.z) * 0.5f};
            const float3 dirs[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
            for ( auto &d : dirs ) {
                Ray ray{c, d, 0.f};
                auto h_bvh = vA.meshes[m].hit(ray, 0.001f, 1e30f,
                                              vA.triangles, vA.mesh_bvh_nodes);
                auto h_lin = vB.meshes[m].hit(ray, 0.001f, 1e30f,
                                              vB.triangles, nullptr);
                bool b = (bool)h_bvh, l = (bool)h_lin;
                if ( b != l ||
                     ( b && (std::fabs(h_bvh->t - h_lin->t) > 1e-4f) ) ) {
                    std::printf("FAIL: mesh[%u] BVH != linear at (%f,%f,%f)+%s "
                                "(bvh=%d lin=%d)\n",
                                m, c.x, c.y, c.z,
                                d.x ? "+X" : d.y ? "+Y" : "+Z", b, l);
                    multi_ok = false;
                }
            }
        }
        std::printf("multi-mesh: BVH == linear scan on %u meshes: %s\n",
                    vA.num_meshes, multi_ok ? "PASS" : "FAIL");
        ok = ok && multi_ok;

        dA.free(&rt);
        dB.free(&rt);
    }

    std::printf(ok ? "OK\n" : "FAILURES\n");
    return ok ? 0 : 1;
}

} // namespace

void register_mesh_diag(argparse::ArgumentParser &diag,
                        std::vector<DiagCommand> &commands) {
    DiagCommand cmd;
    cmd.name = "mesh";
    cmd.parser = std::make_unique<argparse::ArgumentParser>("mesh");
    cmd.parser->add_argument("rays")
        .nargs(argparse::nargs_pattern::optional)
        .default_value(10000)
        .action([](const std::string &s) { return std::stoi(s); })
        .help("number of random rays to test");
    cmd.run = [](argparse::ArgumentParser &p) {
        return run_mesh_diag(p.get<int>("rays"));
    };
    diag.add_subparser(*cmd.parser);
    commands.push_back(std::move(cmd));
}
