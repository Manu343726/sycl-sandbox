#pragma once

#include "diags.h"

/// Register `diag sycl` on `diag`.
void register_sycl_diag(argparse::ArgumentParser &diag,
                        std::vector<DiagCommand> &commands);
