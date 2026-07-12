# Tracy Profiler UI Design Reference

Studied from: https://github.com/wolfpld/tracy (server/profiler UI)

## Timeline Layout (TracyView_Timeline.cpp, TracyView_FrameTimeline.cpp, TracyView_ZoneTimeline.cpp)

1. **DrawTimeline()** — top-level container:
   - `DrawTimelineFramesHeader()` — ruler/scale bar at top
   - `DrawTimelineFrames()` — frame set overview bars (one thin row per frame set)
   - `DrawTimelineSections()` — annotation sections
   - TimelineController manages items (GPU, CPU, threads, plots)

2. **Frame timeline header** (`DrawTimelineFramesHeader`):
   - `ImGui::InvisibleButton("##zoneFrames", ImVec2(w, ty * 1.5f))` as clickable area
   - Ruler with tick marks at log10-scaled intervals
   - Time labels: `TimeToString()` for compact, `TimeToStringExact()` for first
   - Small tick marks at 1/10 intervals between major marks
   - Uses `dpos` (wpos + 0.5) for pixel-perfect alignment

3. **Frame bars** (`DrawTimelineFrames`):
   - One row per frame set, `ty` height
   - Each frame is a colored rect: green (active), white (inactive), red (first)
   - `DrawZigZag` for collapsed frames smaller than `MinFrameSize` (5px)
   - Tooltip on hover shows frame stats

4. **Zone drawing** (`DrawZoneList`, ~670 lines):
   - `TimelineContext` struct passed around: w, ty, sty, scale, yMin, yMax, pxns, nspx, vStart, vEnd, wpos, hover
   - Zone types: `Folded`, `Zone`, `Ghost`, `GhostFolded`
   - Zones at each depth occupy `ostep = ty + 1` pixels
   - `MinVisSize = 3` — minimum visible pixel width

## Zone Text Rendering (TracyView_ZoneTimeline.cpp DrawZoneText lambda)

```cpp
const auto DrawZoneText = [&](uint32_t color, const char* zoneName, ImVec2 tsz,
    double pr0, double pr1, double px0, double px1, double offset) {
    const auto tpx0 = std::max(px0, margin);
    const auto zsz = std::max(pr1 - pr0, pxns * 0.5);
    if (tsz.x < zsz) {
        // Zone wide enough for text — center it
        const auto x = pr0 + (pr1 - pr0 - tsz.x) / 2;
        if (x < margin || x > w - tsz.x) {
            // Would clip — draw clamped and clip-rect
            PushClipRect(...);
            DrawTextContrast(draw, wpos + ImVec2(std::max(tpx0, std::min(double(w - tsz.x), x)), offset), color, zoneName);
            PopClipRect();
        } else {
            DrawTextContrast(draw, wpos + ImVec2(x, offset), color, zoneName);
        }
    } else {
        // Zone too small — draw clipped text
        PushClipRect(wpos + ImVec2(tpx0, offset), wpos + ImVec2(px1, offset + tsz.y * 2), true);
        DrawTextContrast(draw, wpos + ImVec2(tpx0, offset), color, zoneName);
        PopClipRect();
    }
};
```

Also uses `ShortenZoneName()` to strip namespaces/types when text doesn't fit.

## Timing Format (`TimeToString` in TracyPrint.cpp)

- `< 1000 ns`: `"X ns"` (whole number)
- `< 1000 µs`: `"X.XX µs"` (2 fraction digits)
- `< 1000 ms`: `"X.XX ms"`
- `< 60 s`: `"X.XX s"`
- `< 1 h`: `"M:SS.s"`
- `< 24 h`: `"H:MM:SS"`
- `>= 24 h`: `"d H:MM:SS"`

Helper functions: `PrintSmallIntFrac`, `PrintSecondsFrac`, `PrintFrac00`, `PrintFrac0`.
Uses `assert(v < 1000)` for `PrintSmallInt` — very optimized.

## Zone Info Window (TracyView_ZoneInfo.cpp, ~1800 lines)

`DrawZoneInfoWindow()` — separate window (`ImGui::Begin("Zone info", ...)`):
- Zoom to zone button, Go to parent button, Statistics button
- Zone name (big font), function, file:line, thread, execution time
- Self time vs total time
- Time from start of program, wall clock
- **Call stack** section with `TreeNode("Call stack")` — `DrawCallstackTable()`
- **Children list** (`DrawZoneInfoChildren` ~300 lines):
  - "Group children locations" checkbox
  - Self time shown as first entry with progress bar
  - Each child: color box + name, time, individual percentage bar
  - Sortable columns: Zone, Time, MTPC (mean time per call)
  - `PrintStringPercent(buf, TimeToString(time), double(time) * rztime * 100)` + `ProgressBar`

## Find Zone / Distribution Histogram (TracyView_FindZone.cpp, ~2300 lines)

- Separate window, search by name
- Zone list with sortable columns (Start, Time)
- **Histogram**: log/linear toggle, cumulate time toggle
- Hovered bin shows tooltip with time range and count
- Selected zone highlighted with animated marker on histogram
- Distribution statistics: total, mean, median, p75, etc.

## Tooltips

`ZoneTooltip()` (~60 lines):
- Name, function, file:line
- Thread name + ID
- Execution time, self time
- If ghost zone: shows ghost info

## Color Scheme

- `GetThreadColor()` — deterministic hue based on thread ID + depth
- `GetHsvColor()` — for symbols
- `GetSrcLocColor()` — for source locations
- `SmallColorBox()` — tiny colored square before zone names
- `TextFocused(label, value)` — bold label + value on same line
- `TextColoredUnformatted` / `TextDisabledUnformatted`

## Utility

- `DrawZigZag()` — for collapsed small zones (sawtooth pattern)
- `DrawStripedRect()` — striped pattern overlay
- `DrawHistogramMinMaxLabel()` — labels at histogram edges
- `DrawLine()` — custom line drawing to dpos
- `DrawTextContrast()` — text with shadow
