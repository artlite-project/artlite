#ifndef SHIM_XRT_DEVICE_H
#define SHIM_XRT_DEVICE_H

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "shim_xrt/bo.h"
#include "shim_xrt/hwctx.h"

namespace xrt {
class device;
}

namespace shim_xdna {

// XRT places instruction buffers itself: no device heap to reclaim.
inline void set_heap_reclaim(std::function<bool()>) {}

// NPU clocks in MHz (XRT does not report them: 0).
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
  // Data buffers of `sizes` allocated together: ranges (of whole 4 KB pages) of one region of exactly their size, so
  // one XRT buffer that a command lists once, whatever their number and sizes. (user_from: the amdxdna shim's.)
  std::vector<std::unique_ptr<bo>> alloc_bos(std::span<const size_t> sizes, size_t user_from);

  // A context configured by a PDI (opened from an xclbin of it: pdi_xclbin).
  std::unique_ptr<hw_ctx> create_hw_context(std::span<const uint8_t> pdi, std::string_view kernel_name);

  clocks query_clocks() const;
  int power_mode() const { return -1; }  // (xrt-smi reports it)
  std::string driver_version() const { return {}; }
  size_t get_info(uint32_t, std::span<uint8_t>) const {
    throw std::system_error(EOPNOTSUPP, std::generic_category(), "GET_INFO is not available through XRT");
  }
  uint64_t busy_ns() const;
  // No device heap of the process's own: XRT places instruction buffers.
  size_t heap_free() const { return SIZE_MAX; }
  bool heap_fixed() const { return false; }
  bool grow_heap() const { return true; }
  std::optional<amdxdna_async_error> last_async_error() const;

  xrt::device& xrt_device() const { return *m_dev; }

  // The process's live hardware contexts, of any device object (hw_ctx adds and removes itself): a buffer is made in
  // one, of `preferred` if it has one.
  static void add_context(const hw_ctx* ctx);
  static void remove_context(const hw_ctx* ctx);
  static const hw_ctx* any_context(const device* preferred = nullptr);

  // `bytes` (4 KB pages) for a data buffer allocated alone: in the process's current shared region (`offset`), a new
  // one when it is full, or a dedicated region from kSharedMax on. (A command lists at most xrt_abi::kMaxBuffers XRT
  // buffers (abi.h): a program's many buffers reach the NPU through a few regions. A shared region is pinned whole,
  // used or not: what knows its buffers' sizes beforehand allocates them together, alloc_bos.)
  static std::shared_ptr<region> take(size_t bytes, size_t& offset);

  // The process's regions that have an XRT buffer, the largest first (kernel_xrt.cpp: a command lists as many as it has
  // room for besides the regions of its own buffers).
  static std::vector<std::shared_ptr<region>> regions();
  // Counts the regions freed (a region's destructor calls region_gone): a kernel drops the runs it keeps for reuse
  // when it changed, as they hold the XRT buffers of the last command they ran.
  static void region_gone();
  static uint64_t regions_gone();

 private:
  static constexpr size_t kSharedMax = 256ull << 20;
  static constexpr size_t kSharedBytes = 1ull << 30;

  std::shared_ptr<xrt::device> m_dev;  // (the device objects of one index share it)
};

}  // namespace shim_xdna

#endif
