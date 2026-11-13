#include "artlite/scratchpad.h"

#include <fstream>
#include <stdexcept>

namespace artlite {

ParameterScratchpad::ParameterScratchpad(uint32_t* buffer, size_t bufferSizeBytes, const std::string& paramsPath)
    : boMap(buffer) {
  parseParams(paramsPath);
  if (bufferSizeBytes < scratchpadSizeBytes)
    throw std::runtime_error("ParameterScratchpad: buffer size (" +
                             std::to_string(bufferSizeBytes) + ") < required scratchpad size (" +
                             std::to_string(scratchpadSizeBytes) + ")");
  clear();
}

void ParameterScratchpad::writeBits(const std::string& name, uint32_t bits) {
  if (unslotted.count(name))
    throw std::runtime_error("ParameterScratchpad: no command reads parameter '" + name + "', so it has no slot");
  auto it = paramMap.find(name);
  if (it == paramMap.end()) {
    throw std::runtime_error("ParameterScratchpad: unknown parameter '" + name + "'");
  }
  if (auto range = ranges.find(name); range != ranges.end()) {
    int32_t value = static_cast<int32_t>(bits);
    if (value < range->second.first || value > range->second.second)
      throw std::invalid_argument("ParameterScratchpad: value " + std::to_string(value) + " of '" + name +
                                  "' is outside [" + std::to_string(range->second.first) + ", " +
                                  std::to_string(range->second.second) +
                                  "], which keeps its DMA transfers within their buffers");
    values[name] = value;
  }
  uint8_t idx = it->second;
  uint32_t encoded = bits;
  if (coreParams.count(name)) {
    // core parameters require shift-2 to survive masking of lowest bits by
    // firmware op
    encoded = bits << 2;
  } else if (repeatParams.count(name)) {
    encoded = bits << 16;
  }
  boMap[idx] = encoded;
}

void ParameterScratchpad::validate() const {
  for (const JointBound& bound : jointBounds) {
    int64_t offset = values.at(bound.offset);
    int64_t length = values.at(bound.length);
    int64_t end = offset + bound.lengthStep * length;
    if (end > bound.max)
      throw std::invalid_argument("ParameterScratchpad: '" + bound.offset + "' = " + std::to_string(offset) +
                                  " plus " + std::to_string(bound.lengthStep) + " * '" + bound.length +
                                  "' = " + std::to_string(length) + " is " + std::to_string(end) + ", above " +
                                  std::to_string(bound.max) + ", so a DMA transfer using both runs past its buffer");
  }
}

void ParameterScratchpad::clear() {
  for (size_t i = 0; i < scratchpadSizeBytes / 4; i++) {
    boMap[i] = 0;
  }
}

void ParameterScratchpad::parseParams(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("ParameterScratchpad: cannot open '" + path + "'");
  }

  // Format (mlir-aie's emitScratchpadParamsFile):
  //   <num_parameters> <num_slots>
  //   <name> <state_table_idx> <type> <kind> <min> <max>
  //   ...
  //   <num_joint_bounds>
  //   <length> <offset> <length_step> <max>
  //   ...
  //   <num_commands>
  //   <device>:<sequence> <num_read> <name> ...
  //   ...
  // where <state_table_idx> is "-" for a parameter no command reads, kind is
  // "core", "addr" or "repeat", and <min> <max> is "- -" for a parameter with
  // no range. A joint bound requires offset + length_step * length <= max.
  // The commands (runtime sequences no aiex.run calls) may share slots, so a
  // scratchpad holds one command's: an ELF with several is refused.
  unsigned numParams = 0, numSlots = 0;
  file >> numParams >> numSlots;
  if (!file) throw std::runtime_error("ParameterScratchpad: missing parameter and slot counts in '" + path + "'");
  scratchpadSizeBytes = numSlots * 4;

  for (unsigned i = 0; i < numParams; i++) {
    std::string name, slot, type, kind, min, max;
    file >> name >> slot >> type >> kind >> min >> max;
    if (!file)
      throw std::runtime_error("ParameterScratchpad: malformed entry " + std::to_string(i) + " in '" + path + "'");
    if (paramMap.count(name) || unslotted.count(name))
      throw std::runtime_error("ParameterScratchpad: duplicate parameter name '" + name + "'");
    if (slot == "-") {
      unslotted.insert(name);
    } else {
      unsigned long idx = 0;
      try {
        idx = std::stoul(slot);
      } catch (const std::logic_error&) {
        throw std::runtime_error("ParameterScratchpad: malformed slot '" + slot + "' for parameter '" + name +
                                 "' in '" + path + "'");
      }
      if (idx >= numSlots)
        throw std::runtime_error("ParameterScratchpad: state_table_idx " + std::to_string(idx) + " for '" + name +
                                 "' is past the " + std::to_string(numSlots) + " slots of '" + path + "'");
      paramMap[name] = static_cast<uint8_t>(idx);
    }
    if (kind != "core" && kind != "addr" && kind != "repeat") {
      throw std::runtime_error("ParameterScratchpad: invalid kind '" + kind +
                               "' for parameter '" + name + "'");
    } else if (kind == "core") {
      coreParams.insert(name);
    } else if (kind == "repeat") {
      repeatParams.insert(name);
    }
    if (min != "-") {
      try {
        ranges[name] = {std::stoi(min), std::stoi(max)};
      } catch (const std::logic_error&) {
        throw std::runtime_error("ParameterScratchpad: malformed range '" + min + " " + max + "' for parameter '" +
                                 name + "' in '" + path + "'");
      }
      values[name] = 0;
    }
  }

  unsigned numJointBounds = 0;
  file >> numJointBounds;
  if (!file) throw std::runtime_error("ParameterScratchpad: missing joint bound count in '" + path + "'");
  for (unsigned i = 0; i < numJointBounds; i++) {
    JointBound bound;
    file >> bound.length >> bound.offset >> bound.lengthStep >> bound.max;
    if (!file)
      throw std::runtime_error("ParameterScratchpad: malformed joint bound " + std::to_string(i) + " in '" + path +
                               "'");
    for (const std::string& name : {bound.length, bound.offset})
      if (!ranges.count(name))
        throw std::runtime_error("ParameterScratchpad: joint bound " + std::to_string(i) + " names '" + name +
                                 "', which has no range, in '" + path + "'");
    jointBounds.push_back(bound);
  }

  unsigned numCommands = 0;
  file >> numCommands;
  if (!file) throw std::runtime_error("ParameterScratchpad: missing command count in '" + path + "'");
  if (numCommands > 1)
    throw std::runtime_error("ParameterScratchpad: '" + path + "' lists " + std::to_string(numCommands) +
                             " commands, whose parameters may share slots");
}

}  // namespace artlite
