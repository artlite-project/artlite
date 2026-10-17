#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace artlite {

struct KernelEntry {
  std::string name;
  std::vector<uint8_t> pdi;
  // Each element is one runlist entry (instruction stream)
  std::vector<std::vector<uint32_t>> asm_instr_runlist;
};

class Executable {
 public:
  // Load from separate PDI and insts files (one entry, named "MLIR_AIE")
  static Executable from_files(std::string_view pdi_path, std::string_view insts_path);

  const KernelEntry& entry(size_t index) const;
  const KernelEntry& entry(std::string_view name) const;

  ~Executable();
  Executable(Executable&&) noexcept;
  Executable& operator=(Executable&&) noexcept;

 private:
  Executable();
  std::vector<KernelEntry> entries_;
};

}  // namespace artlite
