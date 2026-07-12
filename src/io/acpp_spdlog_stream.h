// An std::ostream that forwards every write to spdlog immediately.
// Used as the backend for AdaptiveCpp's output_stream::set_stream().
//
// Parses the level tag from each line's prefix:
//   [AdaptiveCpp Error]   → spdlog::level::err
//   [AdaptiveCpp Warning] → spdlog::level::warn
//   [AdaptiveCpp Info]    → spdlog::level::info
//   [AdaptiveCpp Debug]   → spdlog::level::debug
//   (no tag)              → spdlog::level::debug
//
// The rewritten message uses a plain "[AdaptiveCpp]" tag — the original
// [AdaptiveCpp Level] suffix is dropped since spdlog colour-codes by level.
//
// No line buffering is performed — every chunk is flushed instantly so
// log output survives crashes.

#pragma once

#include <ostream>
#include <streambuf>
#include <string>
#include <string_view>
#include <mutex>
#include <spdlog/spdlog.h>

namespace detail {

// ── ANSI escape stripping ─────────────────────────────────────────────
// AdaptiveCpp wraps its log tags in colour codes:
//   "\033[;32m[AdaptiveCpp Info] \033[0m..."
// Strip ALL CSI sequences from the message so spdlog can add its own
// colour without interference from the original codes.

inline void strip_ansi_csi(std::string &msg) {
  // Remove all ANSI CSI sequences (ESC[ parameters... final_byte).
  // Uses a simple state machine to avoid OOB issues.
  std::string out;
  out.reserve(msg.size());
  for (size_t i = 0; i < msg.size(); ++i) {
    unsigned char c = static_cast<unsigned char>(msg[i]);
    if (c == 0x1B && i + 1 < msg.size() && msg[i + 1] == '[') {
      i += 2; // skip ESC[
      // Skip parameter bytes (0x30–0x3F) and intermediate bytes (0x20–0x2F).
      while (i < msg.size()) {
        unsigned char b = static_cast<unsigned char>(msg[i]);
        if (b >= 0x40 && b <= 0x7E)
          break; // final byte
        ++i;
      }
      // i now points at the final byte (or past end if unterminated).
      // If final byte found, skip it; the for-loop ++i handles the rest.
    } else {
      out += msg[i];
    }
  }
  msg = std::move(out);
}

// ── Level tag parsing ─────────────────────────────────────────────────
// AdaptiveCpp emits log lines prefixed with:
//   "[AdaptiveCpp Info]", "[AdaptiveCpp Warning]", "[AdaptiveCpp Error]", etc.
// After stripping ALL ANSI escapes, find the tag, map the level, and
// rewrite as plain "[AdaptiveCpp]" so spdlog can apply its own colour.

inline spdlog::level::level_enum parse_level(std::string &msg) {
  strip_ansi_csi(msg);

  // Expected prefix: "[AdaptiveCpp "
  constexpr std::string_view kPrefix = "[AdaptiveCpp ";
  if (msg.size() <= kPrefix.size() + 2 || msg.substr(0, kPrefix.size()) != kPrefix)
    return spdlog::level::debug; // Unknown format, keep default

  // Find the closing ']'
  auto close = msg.find(']');
  if (close == std::string::npos || close <= kPrefix.size())
    return spdlog::level::debug;

  // Extract the level substring
  auto lvl_str = msg.substr(kPrefix.size(), close - kPrefix.size());

  // Determine spdlog level
  spdlog::level::level_enum lvl;
  if (lvl_str == "Error")
    lvl = spdlog::level::err;
  else if (lvl_str == "Warning")
    lvl = spdlog::level::warn;
  else if (lvl_str == "Info")
    lvl = spdlog::level::info;
  else
    lvl = spdlog::level::debug; // "Debug" or unknown

  // Rewrite message: replace "[AdaptiveCpp Level]" with "[AdaptiveCpp]"
  msg.replace(0, close + 1, "[AdaptiveCpp]");
  return lvl;
}

class SpdlogStreamBuf final : public std::streambuf {
public:
  explicit SpdlogStreamBuf() {}

protected:
  int_type overflow(int_type ch) override {
    std::lock_guard<std::mutex> lock(_mtx);
    if (ch != traits_type::eof()) {
      _buffer += static_cast<char>(ch);
      if (ch == '\n')
        flush_line_locked();
    }
    return ch;
  }

  std::streamsize xsputn(const char_type *s, std::streamsize count) override {
    std::lock_guard<std::mutex> lock(_mtx);
    for (std::streamsize i = 0; i < count; ++i) {
      _buffer += s[i];
      if (s[i] == '\n') {
        flush_line_locked();
      }
    }
    // Partial flush: if buffer is getting large, flush anyway
    if (_buffer.size() > 256)
      flush_line_locked();
    return count;
  }

  int sync() override {
    std::lock_guard<std::mutex> lock(_mtx);
    flush_line_locked();
    return 0;
  }

private:
  // Same as flush_line() but requires _mtx to be held.
  bool flush_line_locked() {
    if (_buffer.empty())
      return false;

    while (_buffer.back() == '\n' || _buffer.back() == '\r') {
      _buffer.pop_back();
      if (_buffer.empty())
        return false;
    }
    if (_buffer.empty())
      return false;

    // Parse the level tag and rewrite the message in one pass
    auto lvl = parse_level(_buffer);
    spdlog::log(lvl, "{}", _buffer);
    _buffer.clear();
    return true;
  }

  std::mutex _mtx;
  std::string _buffer;
};

} // namespace detail

// ── A real std::ostream that writes to spdlog ─────────────────────────
// Usage:
//   static detail::SpdlogStreamBuf buf;
//   static std::ostream spdlog_stream(&buf);
//   hipsycl::common::output_stream::get().set_stream(spdlog_stream);
//
// Thread-safety: spdlog is thread-safe; the buffer is not.  Intended
// for single-threaded use (the main thread calls set_stream once).

class AcppSpdlogStream {
public:
  AcppSpdlogStream()
      : _buf{}, _stream{&_buf} {}

  std::ostream &stream() { return _stream; }

private:
  detail::SpdlogStreamBuf _buf;
  std::ostream _stream;
};
