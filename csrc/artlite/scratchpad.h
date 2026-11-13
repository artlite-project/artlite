#pragma once

// The scratchpad of an ELF's runtime parameters (mlir-aie's ``aiex.scratchpad``): a buffer of 32-bit slots the
// firmware's commands read, written by parameter name from the file the compiler emits next to the ELF. A value is
// checked on the host against the range the compiler gave it, and each pair of an offset and a length that one DMA
// transfer uses against its buffer's end, before the NPU runs.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace artlite {

class ParameterScratchpad {
 public:
  /// Construct from a raw buffer and its size in bytes (Python bindings /
  /// testing). The size is required: clear() and every write index off
  /// scratchpadSizeBytes, which parseParams derives from the params file, not
  /// from the buffer.
  ParameterScratchpad(uint32_t* buffer, size_t bufferSizeBytes, const std::string& paramsPath);

  /// Write a raw 32-bit value by name.  For core-kind parameters, the bits
  /// are left-shifted by 2 (firmware requirement); for repeat-kind ones, by
  /// 16 (into the task queue's repeat field).  For addr-kind parameters,
  /// the value is written directly (no shift). A DMA offset or length
  /// parameter's value, read as an int32, must lie in the range the compiler
  /// gave it, which keeps its transfers within their buffers.
  void writeBits(const std::string& name, uint32_t bits);

  /// Write a typed parameter value. For core-kind parameters, the raw bits
  /// are left-shifted by 2 as required by the firmware's UPDATE_REG Incr mode,
  /// and the core right-shifts (unsigned) by 2 after reading. The round trip
  /// therefore zeroes the top 2 bits of the value (the bottom 2 bits are
  /// preserved). Types that fit in 30 bits (e.g. uint16_t, int16_t,
  /// std::bfloat16_t, and uint32_t values < 2^30) round-trip losslessly.
  /// `float` is not supported as a core-kind parameter (the verifier in
  /// `--aie-lower-scratchpad-parameters` rejects it).
  template <typename T>
  void write(const std::string& name, T value) {
    static_assert(sizeof(T) <= 4, "Parameter values must be at most 32 bits");
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    writeBits(name, bits);
  }

  /// Check the written values of each offset and length parameter pair that
  /// one DMA transfer uses. Throws std::invalid_argument if they take the
  /// transfer past the end of its buffer.
  void validate() const;

 private:
  uint32_t* boMap = nullptr;
  size_t scratchpadSizeBytes = 0;
  std::unordered_map<std::string, uint8_t> paramMap;
  std::unordered_set<std::string> unslotted;  // params no command reads
  std::unordered_set<std::string> coreParams;  // params with kind="core"
  std::unordered_set<std::string> repeatParams;  // params with kind="repeat"
  std::unordered_map<std::string, std::pair<int32_t, int32_t>> ranges;
  std::unordered_map<std::string, int32_t> values;  // of the ranged params
  struct JointBound {
    std::string length, offset;
    int64_t lengthStep, max;
  };
  std::vector<JointBound> jointBounds;

  void clear();
  void parseParams(const std::string& path);
};

}  // namespace artlite
