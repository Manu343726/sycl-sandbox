#pragma once
#include "sandbox_api.h"
#include <cstdint>

// Returns true if any param value changed
bool render_param_controls(const KernelDesc& desc, float* params, bool read_only);
