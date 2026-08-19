#pragma once

#include "diags.h"

/// Register `diag tracy` on `diag`.
void register_tracy_diag(argparse::ArgumentParser &diag,
                         std::vector<DiagCommand> &commands);
