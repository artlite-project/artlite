// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
// Modified for Artlite: see LICENSE in this directory.

#ifndef _BO_XDNA_H_
#define _BO_XDNA_H_

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include <string_view>

#include "amdxdna_accel.h"
#include "xrt_mem.h"

namespace shim_xdna {
struct pdev;
struct hw_ctx;

enum xclBOSyncDirection {
  XCL_BO_SYNC_BO_TO_DEVICE = 0,
  XCL_BO_SYNC_BO_FROM_DEVICE,
};

// direction - direction of sync operation
enum class direction {
  host2device = XCL_BO_SYNC_BO_TO_DEVICE,
  device2host = XCL_BO_SYNC_BO_FROM_DEVICE,
};

// properties - buffer details
struct properties {
  shim_xcl_bo_flags flags;  // flags of bo
  uint64_t size;            // size of bo
  uint64_t paddr;           // device address
  uint64_t kmhdl;           // kernel mode handle
};

// Tag of the user-memory constructor.
struct user_memory {};
// Tag of the heap constructor: the CPU address of the heap BO (in its pdev's reservation).
struct heap_memory {
  void* at;
};

// A DRM buffer object, mapped into this process. The NPU's DMA does not snoop the CPU caches:
// sync() writes back and invalidates the range (both directions).
struct bo {
  const pdev& m_pdev;
  void* m_parent = nullptr;  // reserved range of an aligned mapping
  void* m_aligned = nullptr;  // CPU address
  size_t m_parent_size = 0;
  size_t m_aligned_size = 0;
  shim_xcl_bo_flags m_flags{};
  amdxdna_bo_type m_type = AMDXDNA_BO_INVALID;
  uint32_t m_handle = AMDXDNA_INVALID_BO_HANDLE;
  uint64_t m_map_offset = AMDXDNA_INVALID_ADDR;
  uint64_t m_xdna_addr = AMDXDNA_INVALID_ADDR;
  bool m_user = false;  // this process's memory registered as a BO (user_memory)

  // flags: XCL_BO_FLAGS_HOST_ONLY, XRT_BO_FLAGS_CACHEABLE or XCL_BO_FLAGS_EXECBUF
  bo(const pdev& p, size_t size, uint32_t flags);
  // A heap BO of the pdev's device heap (pdev::add_heap_bo), unmapped with the pdev's reservation.
  bo(const pdev& p, size_t size, heap_memory);
  // A host-only BO in this process's memory: anonymous 2 MB pages (MADV_HUGEPAGE, MADV_DONTFORK), faulted in
  // and flushed by several threads, then pinned by the driver (a userptr BO, AMDXDNA_BO_SHARE with a VA table).
  // A SHMEM BO's pages are allocated and mapped by the driver in one thread (16 GB: 1.7 s; this: ~0.3 s).
  bo(const pdev& p, size_t size, user_memory);
  ~bo();
  // no copying
  bo(const bo&) = delete;
  bo& operator=(const bo&) = delete;

  void* map() const;
  void sync(direction, size_t size, size_t offset);
  void sync(direction);
  properties get_properties() const;
  size_t size() const;
  uint32_t get_drm_bo_handle() const;
  // Address the NPU uses (the user VA of the mapping in PASID mode)
  uint64_t get_paddr() const;
  uint64_t paddr_in(const hw_ctx&) const { return get_paddr(); }  // (the same in every hardware context)
  std::string_view type_to_name() const;

 private:
  bo(const pdev& p, size_t size, shim_xcl_bo_flags flags, amdxdna_bo_type type, void* at = nullptr);
  void mmap_bo(void* at);
  void release() noexcept;
};

}  // namespace shim_xdna

#endif
