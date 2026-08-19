#pragma once

#include "diags.h"

/// Register `diag profiler` on `diag` — the full profiler-system
/// diagnostic (device ring → Tracy bridge → client → server capture)
/// across all SYCL backends.
void register_profiler_diag(argparse::ArgumentParser &diag,
                            std::vector<DiagCommand> &commands);