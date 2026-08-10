#pragma once

#include "diags.h"

/// Register `diag stl` and `diag stl-center` on `diag`.
void register_stl_diag(argparse::ArgumentParser &diag,
                       std::vector<DiagCommand> &commands);
