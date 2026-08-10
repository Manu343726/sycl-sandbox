// Central diag registry + dispatch.  Each diag module registers its own
// subcommand(s) into the `diag` parser; this file wires them together
// and runs the one the user selected.
//
// Note: this project uses p-ranav/argparse v3.2 (conan), whose API has
// NO callback support — subcommand dispatch is done manually here.

#include "diags.h"

#include "diags/scene_diag.h"
#include "diags/stl_diag.h"
#include "diags/stl_viz_diag.h"
#include "diags/loader_diag.h"
#include "diags/bvh_diag.h"
#include "diags/mesh_diag.h"
#include "diags/gpu_diag.h"
#include "diags/sycl_diag.h"
#include "diags/camera_diag.h"

#include <cstdio>

namespace {

/// The `diag` parser — lives for the whole process.  Subcommands are
/// registered into it, then it is handed to the program tree via
/// add_subparser().  NOTE: argparse v3.2 stores subparsers by REFERENCE
/// and parses in place on the referenced object — this parser and the
/// per-module subparsers (heap-owned by DiagCommand) must outlive both
/// registration and dispatch.
argparse::ArgumentParser diag_parser("diag");
std::vector<DiagCommand> commands;

} // namespace

void register_diag_subcommands(argparse::ArgumentParser &program) {
    register_scene_diag(diag_parser, commands);
    register_stl_diag(diag_parser, commands);
    register_stl_viz_diag(diag_parser, commands);
    register_loader_diag(diag_parser, commands);
    register_bvh_diag(diag_parser, commands);
    register_mesh_diag(diag_parser, commands);
    register_gpu_diag(diag_parser, commands);
    register_sycl_diag(diag_parser, commands);
    register_camera_diag(diag_parser, commands);
    program.add_subparser(diag_parser);
}

int run_diag_subcommand(argparse::ArgumentParser &program) {
    if (!program.is_subcommand_used("diag"))
        return -1;

    // The parsed diag parser is the same object registered above — values
    // were parsed in place by the program's parse_args().
    auto &diag = program.at<argparse::ArgumentParser>("diag");
    for (const auto &cmd : commands) {
        if (diag.is_subcommand_used(cmd.name)) {
            return cmd.run(diag.at<argparse::ArgumentParser>(cmd.name));
        }
    }
    std::fprintf(stderr, "diag: no subcommand selected\n");
    return 1;
}

