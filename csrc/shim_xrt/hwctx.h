#ifndef SHIM_XRT_HWCTX_H
#define SHIM_XRT_HWCTX_H

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace xrt {
class hw_context;
class kernel;
}  // namespace xrt

namespace shim_xdna {
struct device;

// What `status()` returns on Linux: the driver's entry of a context. XRT reports none of it.
struct amdxdna_drm_hwctx_entry {
  uint32_t context_id, start_col, num_col, state;
  uint64_t command_submissions, command_completions;
  uint32_t txn_op_idx, ctx_pc, fatal_error_type, fatal_error_exception_type, fatal_error_exception_pc,
      fatal_error_app_module;
};

// A hardware context: one xclbin loaded on the NPU, and its kernel.
struct hw_ctx {
  device& m_device;
  std::string m_kernel_name;
  std::unique_ptr<xrt::hw_context> m_ctx;
  std::unique_ptr<xrt::kernel> m_kernel;

  // xclbin: the bytes of the program's .xclbin file.
  hw_ctx(device& dev, std::span<const uint8_t> xclbin, std::string_view kernel_name);
  ~hw_ctx();
  hw_ctx(const hw_ctx&) = delete;
  hw_ctx& operator=(const hw_ctx&) = delete;

  // (the profiler's register reads and the driver's context entry are Linux driver features)
  uint32_t read_aie_reg(uint32_t col, uint32_t row, uint32_t reg_addr) const;
  uint32_t partition_start_col() const;
  amdxdna_drm_hwctx_entry status() const;
};

}  // namespace shim_xdna

#endif
