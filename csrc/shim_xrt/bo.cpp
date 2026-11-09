#include "shim_xrt/bo.h"

#include <emmintrin.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>

#include <cerrno>
#include <system_error>
#endif

#include <algorithm>
#include <atomic>
#include <cstring>
#include <new>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "shim_xrt/device.h"
#include "shim_xrt/hwctx.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

namespace shim_xdna {

namespace {
constexpr size_t kPage = 4096;
size_t round_up(size_t n) { return (std::max<size_t>(n, 1) + kPage - 1) / kPage * kPage; }

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

#ifdef _WIN32
void fault_in(char* p, size_t n) {
  for (size_t o = 0; o < n; o += kPage) p[o] = 0;
}
#else
void fault_in(char* p, size_t n) {
  if (::madvise(p, n, MADV_POPULATE_WRITE))
    for (size_t o = 0; o < n; o += kPage) p[o] = 0;  // (before Linux 5.14)
}
#endif

// Fresh memory faulted in and written back by several threads: no dirty line of the zeroing may later land over what
// the NPU wrote, and the driver's pin would fault the pages in one thread (Linux, 16 GB: ~1.7 s; this: ~0.3 s).
void populate(char* mem, size_t n) {
  constexpr size_t piece = size_t(32) << 20;
  std::atomic<size_t> next{0};
  auto work = [&] {
    for (size_t a; (a = next.fetch_add(piece)) < n;) {
      const size_t k = std::min(piece, n - a);
      fault_in(mem + a, k);
      flush_lines(mem + a, k);
    }
  };
  std::vector<std::thread> pool;
  for (size_t t = 1; t < std::clamp<size_t>(n / (size_t(256) << 20), 1, 8); t++) pool.emplace_back(work);
  work();
  for (auto& th : pool) th.join();
}

#ifdef _WIN32
void* host_alloc(size_t size) {
  const size_t n = round_up(size);
  void* p = ::VirtualAlloc(nullptr, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (!p) throw std::bad_alloc();
  populate(static_cast<char*>(p), n);
  return p;
}

void host_free(void* p, size_t) { ::VirtualFree(p, 0, MEM_RELEASE); }
#else
// Zeroed anonymous memory at a 2 MB boundary (2 MB pages where the kernel has them: fewer page walks for the IOMMU and
// the CPU), kept out of forked children (pinned pages must not turn copy-on-write), populated.
void* host_alloc(size_t size) {
  constexpr size_t huge = size_t(2) << 20;
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
  populate(mem, n);
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
  xrt.reset();
  hold.reset();
  host_free(host, size);
}

xrt::bo& region::xrt_for(const hw_ctx& ctx) {
  std::lock_guard guard(lock);
  if (!xrt) {
    hold = std::make_unique<xrt::hw_context>(*ctx.m_ctx);
    // (XRT makes a buffer only on a device object with a context: the context's)
    xrt = std::make_unique<xrt::ext::bo>(ctx.m_device.xrt_device(), host, size);
  }
  return *xrt;
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
  m_sub.reset();
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

void bo::sync_lines(const int64_t* offsets, size_t count) {
  if (!count) return;
  const auto [lo, hi] = std::minmax_element(offsets, offsets + count);
  if (*lo < 0 || static_cast<uint64_t>(*hi) >= m_size)
    throw std::out_of_range("bo::sync_lines: a line outside the buffer");
  if (m_kind == kind::instructions) {  // (one copy of the lines' span)
    sync(direction::host2device, std::min<size_t>(*hi - *lo + 64, m_size - *lo), static_cast<size_t>(*lo));
    return;
  }
  const char* const host = static_cast<const char*>(m_host);
  _mm_mfence();
  for (size_t i = 0; i < count; i++) _mm_clflush(host + offsets[i]);
  _mm_mfence();
}

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
  flush_lines(static_cast<char*>(m_host) + offset, size);
}

uint64_t bo::get_paddr() {
  if (m_kind == kind::instructions && m_xrt) return m_xrt->address();
  const hw_ctx* ctx = device::any_context(&m_device);
  if (!ctx) throw std::runtime_error("a buffer's device address needs a hardware context: create one first");
  return paddr_in(*ctx);
}

uint64_t bo::paddr_in(const hw_ctx& ctx) {
  if (m_kind == kind::data) return region_for(ctx).address() + m_offset;
  return xrt_for(ctx, 1).address();
}

xrt::bo& bo::region_for(const hw_ctx& ctx) { return m_region->xrt_for(ctx); }

xrt::bo& bo::xrt_for(const hw_ctx& ctx, unsigned arg) {
  if (m_kind == kind::data) {
    xrt::bo& whole = m_region->xrt_for(ctx);
    if (m_offset == 0 && m_region->size == round_up(m_size)) return whole;
    std::lock_guard guard(m_lock);
    if (!m_sub) m_sub = std::make_unique<xrt::bo>(whole, round_up(m_size), m_offset);
    return *m_sub;
  }
  const uint32_t group = ctx.m_kernel->group_id(arg);
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
