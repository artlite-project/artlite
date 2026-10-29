#include "artlite/profiler.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "artlite/error.h"
#include "artlite/shim.h"

namespace artlite {

struct Profiler::Impl {
  shim_xdna::hw_ctx& ctx;
  uint32_t start_col;
  std::vector<std::pair<uint32_t, uint32_t>> tiles;  // design coords
  // (register address, label): core-module counters at 0x31520+4*i,
  // memory-module counters at 0x11020+4*i (max 2 mem counters).
  std::vector<std::pair<uint32_t, std::string>> counters;
  std::vector<std::vector<uint32_t>> prev;
  FILE* file = nullptr;
  bool wrote_event = false;
  bool closed = false;
  std::chrono::steady_clock::time_point t0;

  Impl(shim_xdna::hw_ctx& c, std::vector<std::pair<uint32_t, uint32_t>> t,
       std::vector<std::pair<uint32_t, std::string>> ctrs,
       const std::string& out_path)
      : ctx(c),
        start_col(c.partition_start_col()),
        tiles(std::move(t)),
        counters(std::move(ctrs)),
        prev(tiles.size(), std::vector<uint32_t>(counters.size())),
        t0(std::chrono::steady_clock::now()) {
    if (counters.empty())
      throw artlite::Error("Profiler: no counters configured");
    file = std::fopen(out_path.c_str(), "w");
    if (!file)
      throw artlite::Error("Profiler: cannot open output file '" + out_path +
                           "': " + std::strerror(errno));
    std::fputs("[", file);
    write_metadata();
  }

  ~Impl() { close(); }

  std::vector<std::string> host_tracks;  // record() names, tid = index + 1

  void write_metadata() {
    event("{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":0,\"args\":{\"name\":\"host\"}}");
    for (size_t i = 0; i < tiles.size(); i++) {
      auto [col, row] = tiles[i];
      int pid = static_cast<int>(i + 1);
      event(
          "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":%d,"
          "\"args\":{\"name\":\"tile(%u,%u)\"}}",
          pid, col, row);
      for (size_t c = 0; c < counters.size(); c++)
        event(
            "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,"
            "\"args\":{\"name\":\"%s\"}}",
            pid, static_cast<int>(c + 1), escape(counters[c].second).c_str());
    }
  }

  static std::string escape(std::string_view s) {
    std::string out;
    for (char ch : s) {
      if (ch == '"' || ch == '\\') out.push_back('\\');
      out.push_back(ch);
    }
    return out;
  }

  // The footer ("]") is rewritten after every event so the file is always a
  // valid JSON document, even if the process dies mid-profile.
#if defined(__GNUC__)
  __attribute__((format(printf, 2, 3)))  // (GCC / Clang check each call's arguments against its format)
#endif
  void event(const char* fmt, ...) {
    if (wrote_event) {
      std::fseek(file, -2, SEEK_END);  // step over "\n]"
      std::fputs(",\n", file);
    } else {
      std::fputs("\n", file);
      wrote_event = true;
    }
    std::va_list args;
    va_start(args, fmt);
    std::vfprintf(file, fmt, args);
    va_end(args);
    std::fputs("\n]", file);
    std::fflush(file);
  }

  double now_us() const {
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - t0)
        .count();
  }

  void sample() {
    double ts = now_us();
    for (size_t i = 0; i < tiles.size(); i++) {
      auto [col, row] = tiles[i];
      uint32_t rel_col = col - start_col;
      for (size_t cIdx = 0; cIdx < counters.size(); cIdx++) {
        uint32_t cur = ctx.read_aie_reg(rel_col, row, counters[cIdx].first);
        uint32_t delta = cur - prev[i][cIdx];
        prev[i][cIdx] = cur;
        // One numeric arg only ("value", the cycle/count delta): perfetto
        // renders one track per (name,pid,tid,arg-name), and the tile is
        // already the process. Extra args would spam duplicate tracks.
        event(
            "{\"name\":\"%s\",\"cat\":\"npu_counter\",\"ph\":\"C\","
            "\"ts\":%.3f,\"pid\":%d,\"tid\":%d,"
            "\"args\":{\"value\":%u}}",
            escape(counters[cIdx].second).c_str(), ts, static_cast<int>(i + 1),
            static_cast<int>(cIdx + 1), delta);
      }
    }
  }

  void record(const std::string& name, double value) {
    auto it = std::find(host_tracks.begin(), host_tracks.end(), name);
    int tid = static_cast<int>(it - host_tracks.begin()) + 1;
    if (it == host_tracks.end()) {
      host_tracks.push_back(name);
      event("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":%d,\"args\":{\"name\":\"%s\"}}", tid,
            escape(name).c_str());
    }
    event("{\"name\":\"%s\",\"cat\":\"host\",\"ph\":\"C\",\"ts\":%.3f,\"pid\":0,\"tid\":%d,"
          "\"args\":{\"value\":%.3f}}",
          escape(name).c_str(), now_us(), tid, value);
  }

  void close() {
    if (closed || !file) return;
    if (!wrote_event) std::fputs("]\n", file);
    std::fclose(file);
    file = nullptr;
    closed = true;
  }
};

Profiler::Profiler(shim_xdna::hw_ctx& ctx,
                   std::vector<std::pair<uint32_t, uint32_t>> tiles,
                   std::vector<std::pair<uint32_t, std::string>> counters,
                   std::string out_path)
    : impl_(std::make_unique<Impl>(ctx, std::move(tiles), std::move(counters),
                                   std::move(out_path))) {}

Profiler::~Profiler() {
  if (impl_) impl_->close();
}

void Profiler::sample() { impl_->sample(); }

void Profiler::record(const std::string& name, double value) { impl_->record(name, value); }

void Profiler::close() { impl_->close(); }

void dump_bo(shim_xdna::bo& buf, std::string_view path) {
  buf.sync(shim_xdna::direction::device2host);
  auto size = static_cast<size_t>(buf.get_properties().size);
  const auto* data = static_cast<const uint8_t*>(buf.map());
  FILE* f = std::fopen(std::string(path).c_str(), "wb");
  if (!f)
    throw artlite::Error("dump_bo: cannot open '" + std::string(path) +
                         "': " + std::strerror(errno));
  size_t written = std::fwrite(data, 1, size, f);
  std::fclose(f);
  if (written != size)
    throw artlite::Error("dump_bo: short write to '" + std::string(path) + "'");
}

}  // namespace artlite
