#pragma once

#include "diags.h"

/// Register `diag scene` on `diag` and append its DiagCommand.
void register_scene_diag(argparse::ArgumentParser &diag,
                         std::vector<DiagCommand> &commands);
