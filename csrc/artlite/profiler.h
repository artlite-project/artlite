#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace shim_xdna {
struct bo;
struct hw_ctx;
}  // namespace shim_xdna

namespace artlite {

/// NPU-side profiler for the AIE2P core-module performance counters.
///
/// The counters themselves are programmed by control-code writes injected
/// into the operator's MLIR at compile time (host register writes require
/// root on amdxdna). This class only *reads* the counters after each
/// dispatch through the context-scoped AMDXDNA_AIE_TILE_READ ioctl and
/// appends Chromium-JSON counter ("C") events to the output file.
///
/// Counter values are cumulative between dispatches; sample() emits deltas
/// (modulo 2^32) so each sample corresponds to one dispatch window.
class Profiler {
 public:
  /// tiles: (col, row) of the instrumented core tiles in *design* (absolute)
  /// coordinates. counters: (register address, label) per configured counter;
  /// core-module counters use 0x31520+4*i, memory-module 0x11020+4*i.
  Profiler(shim_xdna::hw_ctx& ctx,
           std::vector<std::pair<uint32_t, uint32_t>> tiles,
           std::vector<std::pair<uint32_t, std::string>> counters,
           std::string out_path);
  ~Profiler();

  Profiler(const Profiler&) = delete;
  Profiler& operator=(const Profiler&) = delete;

  /// Read all counters on all tiles and append counter events. Cheap enough
  /// per dispatch (one firmware round-trip per tile per counter).
  void sample();

  /// Append a host-side counter event (process "host", one track per name):
  /// e.g. a dispatch's wall time, the driver's device busy time, the clock.
  void record(const std::string& name, double value);

  /// Write the closing "]" and flush. Idempotent.
  void close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Sync a buffer device->host and dump its full contents to a file.
/// Used for stream-trace buffers; the dump is a raw little-endian byte
/// image, later converted to Chromium JSON by artlite.profiling.
void dump_bo(shim_xdna::bo& buf, std::string_view path);

}  // namespace artlite
