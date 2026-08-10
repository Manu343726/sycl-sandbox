#pragma once

#include <argparse/argparse.hpp>

#include <vector>

/// Register the `diag mesh` subcommand on `diag`.
void register_mesh_diag(argparse::ArgumentParser &diag,
                        std::vector<struct DiagCommand> &commands);
