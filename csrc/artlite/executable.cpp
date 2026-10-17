#include "artlite/executable.h"

#include <format>
#include <fstream>
#include <utility>

#include "artlite/error.h"

namespace artlite {

Executable::Executable() = default;
Executable::~Executable() = default;
Executable::Executable(Executable&&) noexcept = default;
Executable& Executable::operator=(Executable&&) noexcept = default;

static std::vector<uint8_t> read_binary(std::string_view path) {
  std::ifstream f(std::string(path), std::ios::binary | std::ios::ate);
  if (!f) throw ExecutableError(std::format("Failed to open file: {}", path));
  auto sz = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> data(static_cast<size_t>(sz));
  f.read(reinterpret_cast<char*>(data.data()), sz);
  return data;
}

Executable Executable::from_files(std::string_view pdi_path, std::string_view insts_path) {
  auto pdi_data = read_binary(pdi_path);
  auto insts_data = read_binary(insts_path);

  if (insts_data.size() % sizeof(uint32_t) != 0)
    throw ExecutableError("Instruction file size must be a multiple of 4");

  size_t n_insts = insts_data.size() / sizeof(uint32_t);
  auto* insts_u32 = reinterpret_cast<const uint32_t*>(insts_data.data());

  Executable exe;
  KernelEntry entry;
  entry.name = "MLIR_AIE";
  entry.pdi = std::move(pdi_data);
  entry.asm_instr_runlist.emplace_back(insts_u32, insts_u32 + n_insts);
  exe.entries_.push_back(std::move(entry));
  return exe;
}

const KernelEntry& Executable::entry(size_t index) const {
  if (index >= entries_.size()) throw ExecutableError("Entry index out of range");
  return entries_[index];
}

const KernelEntry& Executable::entry(std::string_view name) const {
  for (auto& e : entries_)
    if (e.name == name) return e;
  throw ExecutableError(std::format("Entry not found: {}", name));
}

}  // namespace artlite
