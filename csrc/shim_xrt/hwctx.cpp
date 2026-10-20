#include "shim_xrt/hwctx.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "shim_xrt/device.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

namespace shim_xdna {

hw_ctx::hw_ctx(device& dev, std::span<const uint8_t> xclbin, std::string_view kernel_name)
    : m_device(dev), m_kernel_name(kernel_name) {
  xrt::device& xdev = dev.xrt_device();
  const std::vector<char> image(reinterpret_cast<const char*>(xclbin.data()),
                                reinterpret_cast<const char*>(xclbin.data()) + xclbin.size());
  const xrt::uuid uuid = xdev.register_xclbin(xrt::xclbin(image));
  m_ctx = std::make_unique<xrt::hw_context>(xdev, uuid);
  m_kernel = std::make_unique<xrt::kernel>(*m_ctx, m_kernel_name);
  dev.add_context(this);
}

hw_ctx::~hw_ctx() { m_device.remove_context(this); }

uint32_t hw_ctx::read_aie_reg(uint32_t, uint32_t, uint32_t) const {
  throw std::runtime_error("reading AIE registers is not available through XRT");
}

uint32_t hw_ctx::partition_start_col() const {
  throw std::runtime_error("the partition's start column is not available through XRT");
}

amdxdna_drm_hwctx_entry hw_ctx::status() const {
  throw std::runtime_error("the context's driver entry is not available through XRT");
}

}  // namespace shim_xdna
