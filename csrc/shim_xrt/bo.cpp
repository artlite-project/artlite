#include "shim_xrt/bo.h"

#include <windows.h>

#include <algorithm>
#include <cstring>
#include <format>
#include <new>
#include <stdexcept>

#include "shim_xrt/device.h"
#include "shim_xrt/hwctx.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

namespace shim_xdna {

namespace {
constexpr size_t kPage = 4096;
size_t round_up(size_t n) { return (std::max<size_t>(n, 1) + kPage - 1) / kPage * kPage; }

void* host_alloc(size_t size) {
  void* p = ::VirtualAlloc(nullptr, round_up(size), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (!p) throw std::bad_alloc();
  return p;
}
}  // namespace

region::region(size_t n) : host(host_alloc(n)), size(round_up(n)) {}

region::~region() {
  by_group.clear();
  ::VirtualFree(host, 0, MEM_RELEASE);
}

xrt::bo& region::xrt_for(const hw_ctx& ctx, uint32_t g) {
  pinned& p = by_group[g];
  if (!p.xrt) {
    p.hold = std::make_unique<xrt::hw_context>(*ctx.m_ctx);
    // (XRT makes a buffer only on a device object with a context: the context's)
    p.xrt = std::make_unique<xrt::bo>(ctx.m_device.xrt_device(), host, size, XRT_BO_FLAGS_HOST_ONLY, g);
  }
  return *p.xrt;
}

bo::bo(device& dev, size_t size, kind k) : m_device(dev), m_size(size), m_kind(k) {
  if (k == kind::data) {
    m_region = dev.take(round_up(size), m_offset);
    m_host = static_cast<char*>(m_region->host) + m_offset;
  } else {
    m_host = host_alloc(size);
  }
}

// (every data buffer is pinned host memory here: a user-memory one is placed like the others)
bo::bo(device& dev, size_t size, user_memory) : bo(dev, size, kind::data) {}

bo::~bo() {
  m_subs.clear();
  m_xrt.reset();
  m_hold.reset();
  if (m_kind == kind::instructions) ::VirtualFree(m_host, 0, MEM_RELEASE);
}

std::string_view bo::type_to_name() const { return m_kind == kind::data ? "data" : "instructions"; }

properties bo::get_properties() const {
  properties p{};
  p.flags.all = m_kind == kind::data ? XCL_BO_FLAGS_HOST_ONLY : XCL_BO_FLAGS_CACHEABLE;
  p.size = m_size;
  return p;
}

void bo::sync(direction dir) { sync(dir, m_size, 0); }

void bo::sync(direction dir, size_t size, size_t offset) {
  if (offset > m_size || size > m_size - offset) throw std::out_of_range("bo::sync: range exceeds the buffer");
  if (m_kind == kind::instructions) {
    if (dir == direction::host2device) {
      m_dirty_lo = std::min(m_dirty_lo, offset);
      m_dirty_hi = std::max(m_dirty_hi, offset + size);
      if (m_xrt) flush();  // (a PDI or scratchpad is never a kernel argument: xrt_for would not copy it)
    } else if (m_xrt) {
      m_xrt->sync(XCL_BO_SYNC_BO_FROM_DEVICE, size, offset);
      std::memcpy(static_cast<char*>(m_host) + offset, m_xrt->map<char*>() + offset, size);
    }
    return;
  }
  if (!m_region->by_group.empty())
    m_region->by_group.begin()->second.xrt->sync(
        dir == direction::host2device ? XCL_BO_SYNC_BO_TO_DEVICE : XCL_BO_SYNC_BO_FROM_DEVICE, size, m_offset + offset);
}

uint64_t bo::get_paddr() {
  if (m_kind == kind::instructions && m_xrt) return m_xrt->address();
  const hw_ctx* ctx = device::any_context(&m_device);
  if (!ctx) throw std::runtime_error("a buffer's device address needs a hardware context: create one first");
  return paddr_in(*ctx);
}

uint64_t bo::paddr_in(const hw_ctx& ctx) {
  if (m_kind == kind::data) return region_for(ctx, 3).address() + m_offset;
  return xrt_for(ctx, 1).address();
}

xrt::bo& bo::region_for(const hw_ctx& ctx, unsigned arg) {
  return m_region->xrt_for(ctx, ctx.m_kernel->group_id(arg));
}

xrt::bo& bo::xrt_for(const hw_ctx& ctx, unsigned arg) {
  if (m_kind == kind::data) {
    xrt::bo& whole = region_for(ctx, arg);
    if (m_offset == 0 && m_region->size == round_up(m_size)) return whole;
    auto& sub = m_subs[ctx.m_kernel->group_id(arg)];
    if (!sub) sub = std::make_unique<xrt::bo>(whole, round_up(m_size), m_offset);
    return *sub;
  }
  const uint32_t group = ctx.m_kernel->group_id(arg);
  if (!m_xrt || m_group != group) {
    m_xrt.reset();
    m_hold = std::make_unique<xrt::hw_context>(*ctx.m_ctx);
    m_xrt = std::make_unique<xrt::bo>(ctx.m_device.xrt_device(), round_up(m_size), XCL_BO_FLAGS_CACHEABLE, group);
    m_dirty_lo = 0;
    m_dirty_hi = m_size;
    m_group = group;
  }
  flush();
  return *m_xrt;
}

void bo::flush() {
  if (m_dirty_lo >= m_dirty_hi) return;
  const size_t n = m_dirty_hi - m_dirty_lo;
  std::memcpy(m_xrt->map<char*>() + m_dirty_lo, static_cast<char*>(m_host) + m_dirty_lo, n);
  m_xrt->sync(XCL_BO_SYNC_BO_TO_DEVICE, n, m_dirty_lo);
  m_dirty_lo = ~size_t(0);
  m_dirty_hi = 0;
}

}  // namespace shim_xdna
