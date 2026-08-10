#pragma once

#include "diags.h"

/// Register `diag orbit-camera` on `diag`.
void register_camera_diag(argparse::ArgumentParser &diag,
                          std::vector<DiagCommand> &commands);
