// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
// Modified for Artlite: see LICENSE in this directory.

#ifndef PCIE_DEVICE_LINUX_XDNA_H
#define PCIE_DEVICE_LINUX_XDNA_H

#include <stddef.h>
#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

#include "amdxdna_accel.h"
#include "bo.h"
#include "hwctx.h"
#include "xrt_mem.h"

namespace shim_xdna {
struct bo;

struct pdev {
  int m_dev_fd = -1;
  std::unique_ptr<bo> m_dev_heap_bo;

  explicit pdev(uint32_t device_index = 0);
  ~pdev();
  pdev(const pdev&) = delete;
  pdev& operator=(const pdev&) = delete;

  // Throws std::system_error with the ioctl's errno.
  void ioctl(unsigned long cmd, void* arg) const;
  // For destructors: returns 0 or the errno.
  int try_ioctl(unsigned long cmd, void* arg) const noexcept;
  void* mmap(void* addr, size_t len, int prot, int flags, off_t offset) const;
};

// NPU clocks in MHz (the driver's PMF reading, else the nominal value of the DPM level).
struct clocks {
  uint32_t mp_npu_mhz;
  uint32_t h_mhz;
};

struct device {
  pdev m_pdev;
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
  // Device busy time of this DRM client (all its contexts), from fdinfo: mailbox send to
  // response, summed over the commands. Reads a /proc file: keep it out of timed regions.
  uint64_t busy_ns() const;
  // The most recent asynchronous AIE error the firmware reported (device-wide), if any. Uses
  // the management mailbox: only after a clean failure, never while a job may be hung.
  std::optional<amdxdna_async_error> last_async_error() const;
};

}  // namespace shim_xdna

#endif
