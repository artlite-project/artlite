// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
// Modified for Artlite: see LICENSE in this directory.

#include "hwctx.h"

#include <drm/drm.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include <cstring>
#include <limits>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "amdxdna_accel.h"
#include "bo.h"
#include "device.h"
#include "shim_err.h"
#include "xrt_mem.h"

namespace {

// GET_ARRAY of a request only xdna-driver's module serves (Linux's own driver: EOPNOTSUPP).
void get_array_xdna_driver(const shim_xdna::pdev& p, amdxdna_drm_get_array& ga, std::string_view what,
                           std::string_view request) {
  try {
    p.ioctl(DRM_IOCTL_AMDXDNA_GET_ARRAY, &ga);
  } catch (const std::system_error& e) {
    if (e.code().value() != EOPNOTSUPP) throw;
    shim_xdna::shim_err(EOPNOTSUPP, "{} needs xdna-driver's amdxdna module (the driver has no {})", what, request);
  }
}

}  // namespace

namespace shim_xdna {

hw_ctx::hw_ctx(device& dev, std::span<const uint8_t> pdi, std::string_view cu_name,
               uint32_t n_rows, uint32_t n_cols)
    : m_device(dev), m_num_rows(n_rows), m_num_cols(n_cols) {
  m_cu_info.push_back({
    .m_name = std::string(cu_name),
    .m_func = /*functional*/ 0,
    .m_pdi = {pdi.begin(), pdi.end()}
  });
  create_ctx_on_device();
  try {
    configure_cus();
  } catch (...) {
    destroy();
    throw;
  }
}

void hw_ctx::configure_cus() {
  size_t num_cus = m_cu_info.size();
  size_t buf_size =
      offsetof(amdxdna_hwctx_param_config_cu, cu_configs) + num_cus * sizeof(amdxdna_cu_config);
  std::vector<char> cu_conf_param_buf(buf_size);
  auto cu_conf_param = reinterpret_cast<amdxdna_hwctx_param_config_cu*>(cu_conf_param_buf.data());

  cu_conf_param->num_cus = static_cast<uint16_t>(num_cus);
  for (size_t i = 0; i < num_cus; i++) {
    cu_info& ci = m_cu_info[i];
    m_pdi_bos.push_back(m_device.alloc_bo(ci.m_pdi.size(), XRT_BO_FLAGS_CACHEABLE));
    std::unique_ptr<bo>& pdi_bo = m_pdi_bos[i];
    std::memcpy(pdi_bo->map(), ci.m_pdi.data(), ci.m_pdi.size());
    pdi_bo->sync(direction::host2device);
    amdxdna_cu_config& cf = cu_conf_param->cu_configs[i];
    cf.cu_bo = pdi_bo->get_drm_bo_handle();
    cf.cu_func = ci.m_func;
  }

  amdxdna_drm_config_hwctx arg = {};
  arg.handle = m_handle;
  arg.param_type = DRM_AMDXDNA_HWCTX_CONFIG_CU;
  arg.param_val = reinterpret_cast<uintptr_t>(cu_conf_param);
  arg.param_val_size = cu_conf_param_buf.size();
  m_device.get_pdev().ioctl(DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, &arg);
}

hw_ctx::~hw_ctx() { destroy(); }

// Destroying a context waits in the driver for its jobs.
void hw_ctx::destroy() noexcept {
  const pdev& p = m_device.get_pdev();
  if (m_handle != AMDXDNA_INVALID_CTX_HANDLE) {
    amdxdna_drm_destroy_hwctx arg = {.handle = m_handle};
    p.try_ioctl(DRM_IOCTL_AMDXDNA_DESTROY_HWCTX, &arg);
    m_handle = AMDXDNA_INVALID_CTX_HANDLE;
  }
  if (m_syncobj != AMDXDNA_INVALID_FENCE_HANDLE) {
    drm_syncobj_destroy dsobj = {.handle = m_syncobj};
    p.try_ioctl(DRM_IOCTL_SYNCOBJ_DESTROY, &dsobj);
    m_syncobj = AMDXDNA_INVALID_FENCE_HANDLE;
  }
}

uint32_t hw_ctx::cu_index(std::string_view cu_name) const {
  for (uint32_t i = 0; i < m_cu_info.size(); i++)
    if (m_cu_info[i].m_name == cu_name) return i;
  shim_err(ENOENT, "CU name ({}) not found", cu_name);
}

void hw_ctx::create_ctx_on_device() {
  amdxdna_qos_info qos = {};  // the driver copies it in: qos_p must be valid
  amdxdna_drm_create_hwctx arg = {};
  arg.qos_p = reinterpret_cast<uintptr_t>(&qos);
  arg.umq_bo = AMDXDNA_INVALID_BO_HANDLE;
  arg.max_opc = 2048;  // (ignored by the AIE2 driver)
  arg.num_tiles = m_num_rows * m_num_cols;
  arg.log_buf_bo = AMDXDNA_INVALID_BO_HANDLE;
  m_device.get_pdev().ioctl(DRM_IOCTL_AMDXDNA_CREATE_HWCTX, &arg);
  m_handle = arg.handle;
  m_syncobj = arg.syncobj_handle;
  if (m_syncobj == AMDXDNA_INVALID_FENCE_HANDLE) {
    destroy();
    shim_err(ENOTSUP, "the driver gave the context no timeline syncobj");
  }
}

uint64_t hw_ctx::submit(const bo& cmd, std::span<const uint32_t> arg_handles) const {
  amdxdna_drm_exec_cmd ecmd = {
      .hwctx = m_handle,
      .type = AMDXDNA_CMD_SUBMIT_EXEC_BUF,
      .cmd_handles = cmd.get_drm_bo_handle(),
      .args = reinterpret_cast<uintptr_t>(arg_handles.data()),
      .cmd_count = 1,
      .arg_count = static_cast<uint32_t>(arg_handles.size()),
  };
  // EINTR: a signal while a 5th command waited for a slot; nothing was queued
  while (::ioctl(m_device.get_pdev().m_dev_fd, DRM_IOCTL_AMDXDNA_EXEC_CMD, &ecmd) == -1)
    if (errno != EINTR) shim_err(errno, "DRM_IOCTL_AMDXDNA_EXEC_CMD IOCTL failed");
  return ecmd.seq;
}

bool hw_ctx::wait(uint64_t seq, uint32_t timeout_ms) const {
  // The deadline is absolute on CLOCK_MONOTONIC, so a wait interrupted by a signal resumes with
  // the same deadline.
  int64_t deadline = std::numeric_limits<int64_t>::max();
  if (timeout_ms) {
    timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    deadline = tp.tv_sec * 1'000'000'000LL + tp.tv_nsec + timeout_ms * 1'000'000LL;
  }
  uint32_t syncobj = m_syncobj;
  drm_syncobj_timeline_wait w = {
      .handles = reinterpret_cast<uintptr_t>(&syncobj),
      .points = reinterpret_cast<uintptr_t>(&seq),
      .timeout_nsec = deadline,
      .count_handles = 1,
      .flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
  };
  while (::ioctl(m_device.get_pdev().m_dev_fd, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &w) == -1) {
    if (errno == ETIME) return false;
    if (errno != EINTR && errno != EAGAIN) shim_err(errno, "DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT IOCTL failed");
  }
  return true;
}

amdxdna_drm_hwctx_entry hw_ctx::status() const {
  // The driver copies min(element_size, its struct size) bytes: an older driver leaves the
  // newer fields zero.
  amdxdna_drm_hwctx_entry e = {};
  e.context_id = m_handle;
  e.pid = getpid();
  amdxdna_drm_get_array ga = {};
  ga.param = DRM_AMDXDNA_HW_CONTEXT_BY_ID;
  ga.element_size = sizeof(e);
  ga.num_element = 1;
  ga.buffer = reinterpret_cast<uintptr_t>(&e);
  get_array_xdna_driver(m_device.get_pdev(), ga, "a context's status", "DRM_AMDXDNA_HW_CONTEXT_BY_ID");
  return e;
}

uint32_t hw_ctx::partition_start_col() const {
  if (m_partition_start_col < 0) m_partition_start_col = static_cast<int>(status().start_col);
  return m_partition_start_col;
}

uint32_t hw_ctx::read_aie_reg(uint32_t col, uint32_t row, uint32_t reg_addr) const {
  amdxdna_drm_aie_tile_access ta{};
  ta.pid = static_cast<uint64_t>(getpid());
  ta.context_id = m_handle;
  ta.col = col;
  ta.row = row;
  ta.addr = reg_addr;
  ta.size = sizeof(uint32_t);
  ta.type = AMDXDNA_AIE_TILE_ACCESS_REG;

  amdxdna_drm_get_array ga{};
  ga.param = DRM_AMDXDNA_AIE_TILE_READ;
  ga.element_size = sizeof(ta);
  ga.num_element = 1;
  ga.buffer = reinterpret_cast<uintptr_t>(&ta);
  get_array_xdna_driver(m_device.get_pdev(), ga, "reading AIE tiles", "DRM_AMDXDNA_AIE_TILE_READ");

  // For REG reads the KMD copies the value over the first 4 bytes of the
  // buffer, which aliases .pid.
  return static_cast<uint32_t>(ta.pid & 0xFFFFFFFFu);
}

}  // namespace shim_xdna
