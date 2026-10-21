#include "shim_xrt/bo.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <emmintrin.h>
#include <sys/mman.h>

#include <atomic>
#include <cerrno>
#include <system_error>
#include <thread>
#include <vector>
#endif

#include <algorithm>
#include <cstring>
#include <new>
#include <span>
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

#ifdef _WIN32
void* host_alloc(size_t size) {
  void* p = ::VirtualAlloc(nullptr, round_up(size), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (!p) throw std::bad_alloc();
  return p;
}

void host_free(void* p, size_t) { ::VirtualFree(p, 0, MEM_RELEASE); }
#else
// Write back and invalidate the cache lines of a range. CLFLUSH is not ordered against younger loads: without the
// fences a read right after a device->host sync can still hit the stale line.
void flush_lines(const void* p, size_t len) {
  if (!len) return;
  constexpr uintptr_t line = 64;
  uintptr_t cur = reinterpret_cast<uintptr_t>(p) & ~(line - 1);
  const uintptr_t end = reinterpret_cast<uintptr_t>(p) + len;
  _mm_mfence();
  for (; cur < end; cur += line) _mm_clflush(reinterpret_cast<const void*>(cur));
  _mm_mfence();
}

// Zeroed anonymous memory at a 2 MB boundary (2 MB pages where the kernel has them: fewer page walks for the IOMMU and
// the CPU), kept out of forked children (pinned pages must not turn copy-on-write), faulted in and written back by
// several threads: no dirty line of the zeroing may later land over what the NPU wrote, and the driver's pin would
// fault the pages in one thread (16 GB: ~1.7 s; this: ~0.3 s).
void* host_alloc(size_t size) {
  constexpr size_t huge = size_t(2) << 20, piece = size_t(32) << 20;
  const size_t n = round_up(size), span = n + huge;
  void* base = ::mmap(nullptr, span, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (base == MAP_FAILED) throw std::bad_alloc();
  char* const lo = static_cast<char*>(base);
  char* const mem = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(lo) + huge - 1) & ~(huge - 1));
  if (mem > lo) ::munmap(lo, mem - lo);
  if (mem + n < lo + span) ::munmap(mem + n, lo + span - (mem + n));
  ::madvise(mem, n, MADV_HUGEPAGE);  // (no THP: 4 KB pages)
  if (::madvise(mem, n, MADV_DONTFORK)) {
    const int err = errno;
    ::munmap(mem, n);
    throw std::system_error(err, std::generic_category(), "madvise(MADV_DONTFORK)");
  }
  std::atomic<size_t> next{0};
  auto populate = [&] {
    for (size_t a; (a = next.fetch_add(piece)) < n;) {
      const size_t k = std::min(piece, n - a);
      if (::madvise(mem + a, k, MADV_POPULATE_WRITE))
        for (size_t o = a; o < a + k; o += kPage) mem[o] = 0;  // (before Linux 5.14)
      flush_lines(mem + a, k);
    }
  };
  std::vector<std::thread> pool;
  for (size_t t = 1; t < std::clamp<size_t>(n / (size_t(256) << 20), 1, 8); t++) pool.emplace_back(populate);
  populate();
  for (auto& th : pool) th.join();
  return mem;
}

void host_free(void* p, size_t size) { ::munmap(p, round_up(size)); }
#endif

// The transaction's operations (header: 4 words, its byte length in the 4th): whether one is a DDR patch (0x81).
// Fixed-size operations by opcode; a block write and the custom operations (0x80 on) carry their byte size (txn.py).
bool has_ddr_patch(std::span<const uint32_t> w) {
  static constexpr uint8_t kWords[] = {6, 0, 0, 7, 7, 1, 1, 7, 4, 0, 4, 0, 3};
  if (w.size() < 4) return false;
  w = w.first(std::min<size_t>(w.size(), w[3] / 4));
  for (size_t p = 4; p < w.size();) {
    const uint32_t op = w[p] & 0xFF;
    if (op == 0x81) return true;
    size_t n = op < std::size(kWords) ? kWords[op] : 0;
    if (op == 1 && p + 3 < w.size()) n = w[p + 3] / 4;
    if (op >= 0x80 && p + 1 < w.size()) n = w[p + 1] / 4;
    if (n == 0) return false;  // (not a transaction: nothing to patch)
    p += n;
  }
  return false;
}
}  // namespace

region::region(size_t n) : host(host_alloc(n)), size(round_up(n)) {}

region::~region() {
  by_group.clear();
  host_free(host, size);
}

xrt::bo& region::xrt_for(const hw_ctx& ctx, uint32_t g) {
  std::lock_guard guard(lock);
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
  if (m_kind == kind::instructions) host_free(m_host, m_size);
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
    std::lock_guard guard(m_lock);
    if (dir == direction::host2device) {
      m_dirty_lo = std::min(m_dirty_lo, offset);
      m_dirty_hi = std::max(m_dirty_hi, offset + size);
      m_patched.reset();
      if (m_xrt) flush();  // (a PDI or scratchpad is never a kernel argument: xrt_for would not copy it)
    } else if (m_xrt) {
      m_xrt->sync(XCL_BO_SYNC_BO_FROM_DEVICE, size, offset);
      std::memcpy(static_cast<char*>(m_host) + offset, m_xrt->map<char*>() + offset, size);
    }
    return;
  }
#ifdef _WIN32
  std::lock_guard guard(m_region->lock);
  if (!m_region->by_group.empty())
    m_region->by_group.begin()->second.xrt->sync(
        dir == direction::host2device ? XCL_BO_SYNC_BO_TO_DEVICE : XCL_BO_SYNC_BO_FROM_DEVICE, size, m_offset + offset);
#else
  flush_lines(static_cast<char*>(m_host) + offset, size);
#endif
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
  const uint32_t group = ctx.m_kernel->group_id(arg);
  if (m_kind == kind::data) {
    xrt::bo& whole = m_region->xrt_for(ctx, group);
    if (m_offset == 0 && m_region->size == round_up(m_size)) return whole;
    std::lock_guard guard(m_lock);
    auto& sub = m_subs[group];
    if (!sub) sub = std::make_unique<xrt::bo>(whole, round_up(m_size), m_offset);
    return *sub;
  }
  std::lock_guard guard(m_lock);
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

bool bo::ddr_patched() {
  std::lock_guard guard(m_lock);
  if (!m_patched)
    m_patched = has_ddr_patch({static_cast<const uint32_t*>(m_host), m_size / sizeof(uint32_t)});
  return *m_patched;
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
