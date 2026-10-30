// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
// Modified for Artlite: see LICENSE in this directory.

#include "bo.h"

#include <drm/drm.h>
#include <emmintrin.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <thread>
#include <vector>

#include "device.h"
#include "shim_err.h"
#include "xrt_mem.h"

namespace {

amdxdna_bo_type flag_to_type(shim_xcl_bo_flags flags) {
  uint32_t boflags = (static_cast<uint32_t>(flags.boflags) << 24);
  switch (boflags) {
    case XCL_BO_FLAGS_NONE:
    case XCL_BO_FLAGS_HOST_ONLY:
      return AMDXDNA_BO_SHMEM;
    case XCL_BO_FLAGS_CACHEABLE:
      return AMDXDNA_BO_DEV;
    case XCL_BO_FLAGS_EXECBUF:
      return AMDXDNA_BO_CMD;
    default:
      break;
  }
  return AMDXDNA_BO_INVALID;
}

// Write back and invalidate the cache lines of a range. CLFLUSH is not ordered against younger
// loads: without the fences a read right after a device->host sync can still hit the stale line
// (the kernel's clflush_cache_range fences the same way). CLFLUSHOPT is no faster on Zen 5.
void clflush_range(const void* p, size_t len) {
  if (!len) return;
  constexpr uintptr_t line = 64;
  uintptr_t cur = reinterpret_cast<uintptr_t>(p) & ~(line - 1);
  uintptr_t end = reinterpret_cast<uintptr_t>(p) + len;
  _mm_mfence();
  for (; cur < end; cur += line) _mm_clflush(reinterpret_cast<const void*>(cur));
  _mm_mfence();
}

}  // namespace

namespace shim_xdna {

std::string_view bo::type_to_name() const {
  switch (m_type) {
    case AMDXDNA_BO_SHMEM:
      return m_user ? "AMDXDNA_BO_SHARE (user memory)" : "AMDXDNA_BO_SHMEM";
    case AMDXDNA_BO_DEV_HEAP:
      return "AMDXDNA_BO_DEV_HEAP";
    case AMDXDNA_BO_DEV:
      return "AMDXDNA_BO_DEV";
    case AMDXDNA_BO_CMD:
      return "AMDXDNA_BO_CMD";
    default:
      return "BO_UNKNOWN";
  }
}

void bo::mmap_bo(void* at) {
  if (m_map_offset == AMDXDNA_INVALID_ADDR) {
    // A DEV BO is a range of the device heap: its CPU address follows from the heap's mapping
    // (the driver's reported user VA exists only in PASID mode).
    m_aligned = m_pdev.dev_bo_address(m_xdna_addr);
    return;
  }

  if (at) {  // (a heap BO, at its place in the pdev's reservation)
    m_aligned = m_pdev.mmap(at, m_aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, m_map_offset);
    return;
  }
  m_aligned = m_pdev.mmap(nullptr, m_aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_map_offset);
}

void bo::release() noexcept {
  if (m_user && m_handle != AMDXDNA_INVALID_BO_HANDLE) {  // (unpinned before the memory goes)
    drm_gem_close close_bo = {m_handle, 0};
    m_pdev.try_ioctl(DRM_IOCTL_GEM_CLOSE, &close_bo);
    m_handle = AMDXDNA_INVALID_BO_HANDLE;
  }
  if (m_map_offset != AMDXDNA_INVALID_ADDR && m_aligned && !m_user && m_type != AMDXDNA_BO_DEV_HEAP)
    ::munmap(m_aligned, m_aligned_size);
  if (m_parent) ::munmap(m_parent, m_parent_size);
  m_aligned = m_parent = nullptr;
  if (m_handle != AMDXDNA_INVALID_BO_HANDLE) {
    drm_gem_close close_bo = {m_handle, 0};
    m_pdev.try_ioctl(DRM_IOCTL_GEM_CLOSE, &close_bo);
    m_handle = AMDXDNA_INVALID_BO_HANDLE;
    if (m_type == AMDXDNA_BO_DEV) m_pdev.dev_bo_released(m_aligned_size);
  }
}

bo::bo(const pdev& pdev, size_t size, shim_xcl_bo_flags flags, amdxdna_bo_type type, void* at)
    : m_pdev(pdev), m_aligned_size(size), m_size(size), m_flags(flags), m_type(type) {
  if (m_type == AMDXDNA_BO_INVALID) shim_err(EINVAL, "Invalid BO flags: {:#x}", flags.all);

  amdxdna_drm_create_bo cbo = {.size = size, .type = static_cast<uint32_t>(m_type)};
  if (m_type == AMDXDNA_BO_DEV)
    m_pdev.create_dev_bo(cbo);
  else
    m_pdev.ioctl(DRM_IOCTL_AMDXDNA_CREATE_BO, &cbo);
  m_handle = cbo.handle;
  try {
    amdxdna_drm_get_bo_info info = {.handle = m_handle};
    m_pdev.ioctl(DRM_IOCTL_AMDXDNA_GET_BO_INFO, &info);
    m_map_offset = info.map_offset;
    m_xdna_addr = info.xdna_addr;
    mmap_bo(at);
  } catch (...) {
    release();
    throw;
  }

  // Newly allocated buffer may contain dirty pages. If used as output buffer,
  // the data in cacheline will be flushed onto memory and pollute the output
  // from device. We perform a cache flush right after the BO is allocated to
  // avoid this issue.
  if (m_type == AMDXDNA_BO_SHMEM) clflush_range(m_aligned, size);
}

bo::bo(const pdev& p, size_t size, user_memory)
    : m_pdev(p), m_aligned_size((size + 4095) & ~size_t(4095)), m_size(size), m_type(AMDXDNA_BO_SHARE), m_user(true) {
  constexpr size_t huge = size_t(2) << 20;
  if (!size) shim_err(EINVAL, "user BO of 0 bytes");
  m_parent_size = (m_aligned_size + huge - 1) / huge * huge + huge;  // (room to align to 2 MB)
  m_parent = ::mmap(nullptr, m_parent_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (m_parent == MAP_FAILED) {
    m_parent = nullptr;
    shim_err(errno, "mmap(len={}) failed", m_parent_size);
  }
  char* mem = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(m_parent) + huge - 1) & ~(huge - 1));
  m_aligned = mem;
  try {
    // (the advice up to the next 2 MB boundary: the buffer's last part is a 2 MB page as well)
    const size_t advised = (m_aligned_size + huge - 1) / huge * huge;
    ::madvise(mem, advised, MADV_HUGEPAGE);  // (no THP: 4 KB pages)
    // no copy of the pinned pages in a forked child (a subprocess started without vfork)
    if (::madvise(mem, advised, MADV_DONTFORK)) shim_err(errno, "madvise(MADV_DONTFORK) failed");
    // fault in (zeroed pages) and write back the zeroes: the NPU's DMA must not find stale DRAM under dirty lines,
    // nor have its output overwritten by their eviction later (as a SHMEM BO's clflush at allocation). In pieces:
    // a populate holds mmap_lock, which other threads' mmap / brk wait for (a tokenizer loading meanwhile: 2x).
    const size_t piece = size_t(32) << 20;
    const unsigned threads = static_cast<unsigned>(std::clamp<size_t>(m_aligned_size / (size_t(256) << 20), 1, 8));
    std::atomic<size_t> next{0};
    std::vector<std::exception_ptr> errors(threads);
    auto populate = [&](unsigned t) {
      try {
        for (size_t a; (a = next.fetch_add(piece)) < m_aligned_size;) {
          size_t n = std::min(piece, m_aligned_size - a);
          if (::madvise(mem + a, n, MADV_POPULATE_WRITE))
            for (size_t o = a; o < a + n; o += 4096) mem[o] = 0;  // (before Linux 5.14)
          clflush_range(mem + a, n);
        }
      } catch (...) {
        errors[t] = std::current_exception();
      }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < threads; t++) pool.emplace_back(populate, t);
    populate(0);
    for (auto& th : pool) th.join();
    for (auto& e : errors)
      if (e) std::rethrow_exception(e);

    std::vector<uint8_t> tbl(sizeof(amdxdna_drm_va_tbl) + sizeof(amdxdna_drm_va_entry));
    amdxdna_drm_va_tbl head = {.dmabuf_fd = -1, .num_entries = 1};
    amdxdna_drm_va_entry entry = {.vaddr = reinterpret_cast<uintptr_t>(mem), .len = m_aligned_size};
    std::memcpy(tbl.data(), &head, sizeof(head));
    std::memcpy(tbl.data() + sizeof(head), &entry, sizeof(entry));
    amdxdna_drm_create_bo cbo = {.vaddr = reinterpret_cast<uintptr_t>(tbl.data()), .size = m_aligned_size,
                                 .type = static_cast<uint32_t>(AMDXDNA_BO_SHARE)};
    if (::ioctl(m_pdev.m_dev_fd, DRM_IOCTL_AMDXDNA_CREATE_BO, &cbo) == -1) {
      int err = errno;
      shim_err(err, "a user BO of {} MB: DRM_IOCTL_AMDXDNA_CREATE_BO failed{}", m_aligned_size >> 20,
               err == ENOMEM || err == EPERM ? pdev::memlock_hint() : "");  // (EPERM: a limit of 0)
    }
    m_handle = cbo.handle;
    amdxdna_drm_get_bo_info info = {.handle = m_handle};
    m_pdev.ioctl(DRM_IOCTL_AMDXDNA_GET_BO_INFO, &info);
    m_xdna_addr = info.xdna_addr;
  } catch (...) {
    release();
    throw;
  }
}

bo::bo(const pdev& p, size_t size, uint32_t flags)
    : bo(p, size, shim_xcl_bo_flags{.flags = flags}, flag_to_type(shim_xcl_bo_flags{.flags = flags})) {}

bo::bo(const pdev& p, size_t size, heap_memory h) : bo(p, size, shim_xcl_bo_flags{}, AMDXDNA_BO_DEV_HEAP, h.at) {}

// GEM_CLOSE of a BO a job still uses is safe for the driver (it holds its own references) but
// not for the job: the caller keeps BOs alive until their commands complete.
bo::~bo() { release(); }

properties bo::get_properties() const {
  return {m_flags, m_aligned_size, get_paddr(), get_drm_bo_handle()};
}

size_t bo::size() const { return m_size; }

void* bo::map() const { return m_aligned; }

uint64_t bo::get_paddr() const {
  if (m_xdna_addr != AMDXDNA_INVALID_ADDR) return m_xdna_addr;
  return reinterpret_cast<uintptr_t>(m_aligned);
}

uint32_t bo::get_drm_bo_handle() const { return m_handle; }

// Both directions flush the range: CLFLUSH writes dirty lines back and invalidates them.
void bo::sync(direction, size_t size, size_t offset) {
  if (offset > m_aligned_size || size > m_aligned_size - offset)
    shim_err(EINVAL, "Invalid BO offset and size for sync'ing: {}, {}", offset, size);

  switch (m_type) {
    case AMDXDNA_BO_SHMEM:
    case AMDXDNA_BO_CMD:
    case AMDXDNA_BO_DEV:
      clflush_range(static_cast<char*>(m_aligned) + offset, size);
      break;
    default:
      shim_err(ENOTSUP, "Can't sync bo type {}", static_cast<int>(m_type));
  }
}

void bo::sync(direction dir) { sync(dir, size(), 0); }

void bo::sync_lines(const int64_t* offsets, size_t count) {
  _mm_mfence();
  for (size_t i = 0; i < count; i++) _mm_clflush(static_cast<const char*>(m_aligned) + offsets[i]);
  _mm_mfence();
}

}  // namespace shim_xdna
