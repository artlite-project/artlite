// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
// Modified for Artlite: see LICENSE in this directory.

#ifndef _HWCTX_XDNA_H_
#define _HWCTX_XDNA_H_

#include <stddef.h>
#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "amdxdna_accel.h"
#include "xrt_mem.h"

namespace shim_xdna {

struct bo;
struct device;

struct cu_info {
  std::string m_name;
  size_t m_func;
  std::vector<uint8_t> m_pdi;
};

// A hardware context: one timeline of commands (at most 4 in flight in the driver; a 5th submit
// blocks until one completes). Its CU is configured from one PDI (CONFIG_CU).
struct hw_ctx {
  device& m_device;
  uint32_t m_handle = AMDXDNA_INVALID_CTX_HANDLE;
  uint32_t m_syncobj = AMDXDNA_INVALID_FENCE_HANDLE;
  std::vector<cu_info> m_cu_info;
  uint32_t m_num_rows;
  uint32_t m_num_cols;
  std::vector<std::unique_ptr<bo>> m_pdi_bos;

  hw_ctx(device& dev, std::span<const uint8_t> pdi, std::string_view cu_name, uint32_t n_rows,
         uint32_t n_cols);
  ~hw_ctx();
  // no copying
  hw_ctx(const hw_ctx&) = delete;
  hw_ctx& operator=(const hw_ctx&) = delete;

  uint32_t cu_index(std::string_view cu_name) const;

  // Submit an exec-buffer command; returns its point on the context's timeline. arg_handles:
  // BOs the command uses (for the drivers that pin a command's BOs for the job; others ignore them).
  uint64_t submit(const bo& cmd, std::span<const uint32_t> arg_handles) const;

  // Wait until every command up to `seq` finished (the driver writes a command's state before
  // it signals its point). False on timeout (timeout_ms 0: no timeout); a timed-out command
  // keeps running, there is no way to cancel it.
  bool wait(uint64_t seq, uint32_t timeout_ms) const;

  /// Read a 32-bit tile-local register of an AIE tile inside this context's
  /// partition through the firmware (AMDXDNA_AIE_TILE_READ).
  /// col is relative to the partition (see partition_start_col()).
  uint32_t read_aie_reg(uint32_t col, uint32_t row, uint32_t reg_addr) const;

  /// The starting column of this context's partition on the device. The KMD
  /// may place a context at a non-zero column; design-absolute tile columns
  /// are partition-relative by `col - partition_start_col()`.
  uint32_t partition_start_col() const;

  /// The driver's entry of this context (submissions, completions, and the firmware's health
  /// report: txn_op_idx, ctx_pc, fatal error fields). A firmware query through the management
  /// mailbox: use it after a clean failure, never while a job may be hung (a mailbox timeout
  /// destroys the management channel).
  amdxdna_drm_hwctx_entry status() const;

 private:
  void create_ctx_on_device();
  void configure_cus();
  void destroy() noexcept;
  mutable int m_partition_start_col = -1;
};

}  // namespace shim_xdna

#endif  // _HWCTX_XDNA_H_
