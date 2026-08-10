#pragma once

#include "diags.h"

/// Register `diag bvh` on `diag`.
void register_bvh_diag(argparse::ArgumentParser &diag,
                       std::vector<DiagCommand> &commands);
