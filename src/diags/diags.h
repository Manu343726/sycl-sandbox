// sycl-sandbox — Diagnostic subcommands
//
// Every diagnostic that used to live as a throwaway /tmp harness is
// registered here as a `sycl-sandbox diag <subcommand>` CLI entry point:
//
//   diag scene [yaml] [--strict]
//                       Load a YAML scene in software mode, print handle /
//                       triangle / BVH / light counts, mesh windows with
//                       transformed AABBs and materials, canned ray hits.
//                       --strict enforces the mesh_demo.yaml expectations.
//   diag stl [file]     Analyze an STL file with the project loader:
//                       triangle count, AABB, degenerate (zero-area) tris.
//   diag stl-center <in> <out>
//                       Re-center an STL (origin-centered X/Z, bottom at
//                       y=0) and write a clean binary STL.
//   diag stl-viz <file> <outprefix> [--rx] [--ry] [--rz]
//                       Render an STL to shaded orthographic PNGs
//                       (front/side/top/three-quarter) to inspect its true
//                       orientation; --rx/--ry/--rz apply a candidate Euler
//                       rotation (Z→Y→X, same as scene YAML) via the project
//                       loader so a fix can be verified before committing.
//                       PNG via stb_image_write (FetchContent); software
//                       z-buffered rasterizer, no GL — runs headless.
//   diag loader [yaml]  Dump the scene descriptor (params, data sources,
//                       initial camera) and the built scene arrays.
//   diag bvh [n]        BVH traversal vs linear scan equivalence check on
//                       a synthetic Cornell-box scene with random rays.
//   diag gpu [yaml] [--so] [--width] [--height] [--frames]
//                       Drive the REAL kernel .so (dlopen) on the SYCL GPU
//                       backend like the app does, render N frames and
//                       report NaN/Inf pixel counts + average luminance.
//   diag sycl           SYCL runtime probe: platform/device enumeration,
//                       in-order GPU queue creation + device info queries.
//                       (Host-compiled code cannot dispatch kernels — the
//                       acpp device pass registers launchers only in the
//                       kernel .so — so kernel execution is covered by
//                       `diag gpu`.)
//   diag orbit-camera   Verify the scene-debug orbit camera placement math
//                       (inverse of orbit_eye()) is exact for sample poses.
//
// These run before any GLFW/ImGui/app state is created and exit with
// 0 on success, non-zero on failure.

#pragma once

#include <argparse/argparse.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

/// One diagnostic subcommand of the `diag` tree.
struct DiagCommand {
    /// Subcommand name (matches the subparser's program name).
    std::string name;
    /// Heap-owned subparser.  argparse stores subparsers by REFERENCE
    /// (std::reference_wrapper) and parses in place, so the parser must
    /// outlive both registration and dispatch.
    std::unique_ptr<argparse::ArgumentParser> parser;
    /// Runner invoked by the dispatcher with the *parsed* subparser.
    /// Returns the process exit code.
    std::function<int(argparse::ArgumentParser &)> run;
};

/// Register the `diag` subcommand tree on `program`.
void register_diag_subcommands(argparse::ArgumentParser &program);

/// Dispatch to the selected diag subcommand (called after parsing).
/// Returns the process exit code, or -1 if `diag` was not used.
int run_diag_subcommand(argparse::ArgumentParser &program);

