#pragma once

#include "diags.h"

/// Register `diag gpu` on `diag`.
void register_gpu_diag(argparse::ArgumentParser &diag,
                       std::vector<DiagCommand> &commands);
