#pragma once

#include "diags.h"

/// Register `diag loader` on `diag`.
void register_loader_diag(argparse::ArgumentParser &diag,
                          std::vector<DiagCommand> &commands);
