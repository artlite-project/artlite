#include "shim_xrt/hwctx.h"

#include <cerrno>
#include <system_error>
#include <vector>

#include "shim_xrt/device.h"
#include "shim_xrt/xclbin.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

namespace shim_xdna {

namespace {
[[noreturn]] void not_on_xrt(const char* what) {
  throw std::system_error(EOPNOTSUPP, std::generic_category(), std::string(what) + " is not available through XRT");
}
}  // namespace

hw_ctx::hw_ctx(device& dev, std::span<const uint8_t> pdi, std::string_view kernel_name)
    : m_device(dev), m_kernel_name(kernel_name) {
  xrt::device& xdev = dev.xrt_device();
  const xrt::uuid uuid = xdev.register_xclbin(xrt::xclbin(pdi_xclbin(pdi)));
  m_ctx = std::make_unique<xrt::hw_context>(xdev, uuid);
  m_kernel = std::make_unique<xrt::kernel>(*m_ctx, kXclbinKernel);
  dev.add_context(this);
}

hw_ctx::~hw_ctx() { m_device.remove_context(this); }

uint32_t hw_ctx::read_aie_reg(uint32_t, uint32_t, uint32_t) const { not_on_xrt("reading AIE registers"); }

uint32_t hw_ctx::partition_start_col() const { not_on_xrt("the partition's start column"); }

amdxdna_drm_hwctx_entry hw_ctx::status() const { not_on_xrt("the context's driver entry"); }

}  // namespace shim_xdna
