#ifndef SHIM_XRT_DEVICE_H
#define SHIM_XRT_DEVICE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "shim_xrt/bo.h"
#include "shim_xrt/hwctx.h"

namespace xrt {
class device;
}

namespace shim_xdna {

// NPU clocks in MHz (XRT does not report them).
struct clocks {
  uint32_t mp_npu_mhz;
  uint32_t h_mhz;
};

struct amdxdna_async_error {
  uint64_t err_code, ts_us, ex_err_code;
};

struct device {
  explicit device(uint32_t device_index = 0);
  ~device();
  device(const device&) = delete;
  device& operator=(const device&) = delete;

  // flags: XCL_BO_FLAGS_CACHEABLE makes an instruction buffer, any other a data buffer.
  std::unique_ptr<bo> alloc_bo(size_t size, uint32_t flags);
  std::unique_ptr<bo> alloc_user_bo(size_t size);

  // xclbin: the bytes of the program's .xclbin file.
  std::unique_ptr<hw_ctx> create_hw_context(std::span<const uint8_t> xclbin, std::string_view kernel_name);

  clocks query_clocks() const;
  uint64_t busy_ns() const;
  std::optional<amdxdna_async_error> last_async_error() const;

  xrt::device& xrt_device() const { return *m_dev; }

  // The process's live hardware contexts, of any device object (hw_ctx adds and removes itself): a buffer takes its
  // memory groups from one, of `preferred` if it has one.
  static void add_context(const hw_ctx* ctx);
  static void remove_context(const hw_ctx* ctx);
  static const hw_ctx* any_context(const device* preferred = nullptr);

  // `bytes` (4 KB pages) for a data buffer: in the process's current shared region (`offset`), a new one when it is
  // full, a region of its own from kSharedMax. (A command lists at most 5 XRT buffers: a prefill's dozen buffers
  // besides the weights, Qwen3.8 27B's KV cache and GDN states among them, reach the NPU through one or two regions.)
  static std::shared_ptr<region> take(size_t bytes, size_t& offset);

 private:
  static constexpr size_t kSharedMax = 256ull << 20;
  static constexpr size_t kSharedBytes = 1ull << 30;

  std::unique_ptr<xrt::device> m_dev;
};

}  // namespace shim_xdna

#endif
