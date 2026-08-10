#pragma once

#include "diags.h"

/// Register `diag stl-viz` on `diag`.
void register_stl_viz_diag(argparse::ArgumentParser &diag,
                           std::vector<DiagCommand> &commands);
