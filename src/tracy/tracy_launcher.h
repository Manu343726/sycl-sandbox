#pragma once

#ifdef SANDBOX_ENABLE_TRACY

#include <cstdint>
#include <string>

namespace tracy_launcher {

/// Launch the standalone tracy-profiler executable detached from the
/// sandbox process.  The profiler connects to the in-process Tracy
/// client — see docs/architecture.md.
///
/// @param build_dir  Build directory the app was launched from (build/ or
///                   build_debug/).
/// @param addr       Address to auto-connect to (empty = manual connect).
/// @param port       Port number (default 8086).
/// @return true if the executable was found and spawned.
bool launch_profiler(const std::string &build_dir,
                     const std::string &addr = "",
                     uint16_t port = 8086);

} // namespace tracy_launcher

#endif // SANDBOX_ENABLE_TRACY
