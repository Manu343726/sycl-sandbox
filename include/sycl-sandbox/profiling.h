#pragma once

/// Thin wrapper around Tracy profiler macros.
///
/// Include this header in any file that needs profiling zones.  When
/// TRACY_ENABLE is defined (via CMake option -DTRACY_PROFILER=ON), the macros
/// expand to Tracy's instrumented versions.  Otherwise they are no-ops with
/// zero overhead.

#ifdef TRACY_ENABLE

#include <tracy/Tracy.hpp>

#define PROF_FRAME_MARK            FrameMark
#define PROF_FRAME_MARK_N(name)    FrameMarkNamed(name)
#define PROF_ZONE_SCOPED           ZoneScoped
#define PROF_ZONE_SCOPED_N(name)   ZoneNamedN(name, true)

#else

#define PROF_FRAME_MARK
#define PROF_FRAME_MARK_N(name)
#define PROF_ZONE_SCOPED
#define PROF_ZONE_SCOPED_N(name)

#endif
