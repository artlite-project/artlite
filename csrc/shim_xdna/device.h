// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
// Modified for Artlite: see LICENSE in this directory.

#ifndef PCIE_DEVICE_LINUX_XDNA_H
#define PCIE_DEVICE_LINUX_XDNA_H

#include <stddef.h>
#include <sys/types.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "amdxdna_accel.h"
#include "bo.h"
#include "hwctx.h"
#include "xrt_mem.h"

namespace shim_xdna {
struct bo;

// An open DRM file of the device. The driver resolves a command's BO handles in the file it is
// submitted to, so a process opens one per device (open()): every device of an index shares it.
struct pdev {
  int m_dev_fd = -1;
  // The device heap (DEV BOs are ranges of it): heap BOs of 64 MB mapped one after the other into
  // a reservation of the driver's maximum, as the driver wants the heap contiguous in the NPU's and
  // the CPU's addresses. It grows when a DEV BO does not fit, unless the driver refuses a second
  // heap BO (Linux before 7.2: one 64 MB heap per process; m_heap_fixed).
  void* m_heap_range = nullptr;  // the reservation, m_heap_range_size bytes
  size_t m_heap_range_size = 0;
  char* m_heap_cpu = nullptr;  // the heap's CPU address (64 MB aligned in the reservation)
  uint64_t m_heap_xdna_addr = AMDXDNA_INVALID_ADDR;  // and its device address
  mutable std::vector<std::unique_ptr<bo>> m_heap_bos;  // (m_heap_lock)
  // The heap the process uses as the driver spends it: each DEV BO rounded up to 32 KB, 128 KB per
  // hardware context (its 4 command buffers); and whether the driver refused a heap BO.
  mutable size_t m_heap_used = 0;  // (m_heap_lock)
  mutable bool m_heap_fixed = false;  // (m_heap_lock)
  mutable std::mutex m_heap_lock;

  explicit pdev(uint32_t device_index = 0);
  ~pdev();
  pdev(const pdev&) = delete;
  pdev& operator=(const pdev&) = delete;

  // The process's pdev of a device: opened by its first user, closed with its last.
  static std::shared_ptr<pdev> open(uint32_t device_index);

  // Throws std::system_error with the ioctl's errno.
  void ioctl(unsigned long cmd, void* arg) const;
  // ", as ulimit -l is N KB ..." when the locked-memory limit is finite, else "": the driver pins a
  // user BO's pages within it, and a MAP_LOCKED mapping is locked within it.
  static std::string memlock_hint();
  // For destructors: returns 0 or the errno.
  int try_ioctl(unsigned long cmd, void* arg) const noexcept;
  void* mmap(void* addr, size_t len, int prot, int flags, off_t offset) const;
  // CREATE_BO of a DEV BO, adding heap BOs while it does not fit; a full heap throws ENOSPC.
  void create_dev_bo(amdxdna_drm_create_bo& cbo) const;
  // The heap's accounting: a DEV BO of `size` bytes released; a hardware context created (+1) or
  // destroyed (-1).
  void dev_bo_released(size_t size) const;
  void context_heap(int delta) const;
  // Bytes of the heap BOs the process does not use yet (m_heap_used).
  size_t heap_free() const;
  // The driver refused a heap BO: the heap cannot grow.
  bool heap_fixed() const;
  // Adds a heap BO; false where the driver cannot grow the heap (then heap_fixed) or at its maximum.
  bool grow_heap() const;
  // Called, without the heap's lock, when a DEV BO does not fit a heap that cannot grow: frees DEV BOs the process
  // can spare (true: the BO is tried once more). One for the process (set_heap_reclaim).
  inline static std::function<bool()> heap_reclaim;
  // The CPU address of a DEV BO at device address xdna_addr.
  void* dev_bo_address(uint64_t xdna_addr) const;

 private:
  void add_heap_bo() const;
  // add_heap_bo (m_heap_lock held); false where the driver refuses one (m_heap_fixed set) or at the maximum.
  bool try_add_heap_bo() const;
};

inline void set_heap_reclaim(std::function<bool()> f) { pdev::heap_reclaim = std::move(f); }

// NPU clocks in MHz (the driver's PMF reading, else the nominal value of the DPM level).
struct clocks {
  uint32_t mp_npu_mhz;
  uint32_t h_mhz;
};

struct device {
  std::shared_ptr<pdev> m_pdev;
  uint32_t n_rows;
  uint32_t n_cols;

  explicit device(uint32_t device_index = 0);
  ~device();

  const pdev& get_pdev() const;

  std::unique_ptr<bo> alloc_bo(size_t size, uint32_t flags);
  // A host-only BO in this process's memory (bo's user_memory constructor): for large buffers.
  std::unique_ptr<bo> alloc_user_bo(size_t size);

  std::unique_ptr<hw_ctx> create_hw_context(std::span<const uint8_t> pdi,
                                            std::string_view cu_name);

  clocks query_clocks() const;
  // The NPU's power mode (enum amdxdna_power_mode_type).
  int power_mode() const;
  // The DRM driver's name and version ("amdxdna_accel_driver 0.6.0": Linux 7.0's).
  std::string driver_version() const;
  // Device busy time of this DRM client (all the process's contexts on the device), from fdinfo:
  // mailbox send to response, summed over the commands; 0 from a driver without the fdinfo line
  // (Linux's own driver). Reads a /proc file: keep it out of timed regions.
  uint64_t busy_ns() const;
  // The process's device heap (pdev): bytes not in use yet, whether it cannot grow, add a heap BO.
  size_t heap_free() const { return m_pdev->heap_free(); }
  bool heap_fixed() const { return m_pdev->heap_fixed(); }
  bool grow_heap() const { return m_pdev->grow_heap(); }
  // The most recent asynchronous AIE error the firmware reported (device-wide), if any. Uses
  // the management mailbox: only after a clean failure, never while a job may be hung.
  std::optional<amdxdna_async_error> last_async_error() const;
};

}  // namespace shim_xdna

#endif
