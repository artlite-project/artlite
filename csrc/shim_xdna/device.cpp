// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
// Modified for Artlite: see LICENSE in this directory.

#include "device.h"

#include <drm/drm.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <charconv>
#include <cstdio>
#include <fstream>
#include <string>

#include "bo.h"
#include "hwctx.h"
#include "shim_err.h"
#include "xrt_mem.h"

namespace {

std::string ioctl_cmd2name(unsigned long cmd) {
  switch (cmd) {
    case DRM_IOCTL_AMDXDNA_CREATE_HWCTX:
      return "DRM_IOCTL_AMDXDNA_CREATE_HWCTX";
    case DRM_IOCTL_AMDXDNA_DESTROY_HWCTX:
      return "DRM_IOCTL_AMDXDNA_DESTROY_HWCTX";
    case DRM_IOCTL_AMDXDNA_CONFIG_HWCTX:
      return "DRM_IOCTL_AMDXDNA_CONFIG_HWCTX";
    case DRM_IOCTL_AMDXDNA_CREATE_BO:
      return "DRM_IOCTL_AMDXDNA_CREATE_BO";
    case DRM_IOCTL_AMDXDNA_GET_BO_INFO:
      return "DRM_IOCTL_AMDXDNA_GET_BO_INFO";
    case DRM_IOCTL_AMDXDNA_EXEC_CMD:
      return "DRM_IOCTL_AMDXDNA_EXEC_CMD";
    case DRM_IOCTL_AMDXDNA_GET_INFO:
      return "DRM_IOCTL_AMDXDNA_GET_INFO";
    case DRM_IOCTL_AMDXDNA_GET_ARRAY:
      return "DRM_IOCTL_AMDXDNA_GET_ARRAY";
    case DRM_IOCTL_GEM_CLOSE:
      return "DRM_IOCTL_GEM_CLOSE";
    case DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT:
      return "DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT";
    case DRM_IOCTL_SYNCOBJ_DESTROY:
      return "DRM_IOCTL_SYNCOBJ_DESTROY";
    default:
      return "UNKNOWN(" + std::to_string(cmd) + ")";
  }
}

// Device memory heap needs to be within one 64MB page. The maximum size is
// 64MB.
const size_t dev_mem_size = (64 << 20);
}  // namespace

namespace shim_xdna {

pdev::pdev(uint32_t device_index) {
  std::string path = "/dev/accel/accel" + std::to_string(device_index);
  m_dev_fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
  if (m_dev_fd < 0) shim_err(errno, "Failed to open device {}", path);
  try {
    m_dev_heap_bo = std::make_unique<bo>(*this, dev_mem_size, AMDXDNA_BO_DEV_HEAP);
  } catch (...) {
    ::close(m_dev_fd);
    throw;
  }
}

pdev::~pdev() {
  m_dev_heap_bo.reset();
  ::close(m_dev_fd);
}

void pdev::ioctl(unsigned long cmd, void* arg) const {
  if (::ioctl(m_dev_fd, cmd, arg) == -1) shim_err(errno, "{} IOCTL failed", ioctl_cmd2name(cmd));
}

int pdev::try_ioctl(unsigned long cmd, void* arg) const noexcept {
  if (::ioctl(m_dev_fd, cmd, arg) == 0) return 0;
  int err = errno;
  std::fprintf(stderr, "shim_xdna: %s IOCTL failed: errno %d\n", ioctl_cmd2name(cmd).c_str(), err);
  return err;
}

void* pdev::mmap(void* addr, size_t len, int prot, int flags, off_t offset) const {
  void* ret = ::mmap(addr, len, prot, flags, m_dev_fd, offset);
  if (ret == MAP_FAILED)
    shim_err(errno, "mmap(addr={}, len={}, prot={}, flags={}, offset={}) failed", addr, len, prot,
             flags, offset);
  return ret;
}

template <typename T>
static T get_info(const pdev& dev, uint32_t param) {
  T value = {};
  amdxdna_drm_get_info arg = {.param = param,
                              .buffer_size = sizeof(value),
                              .buffer = reinterpret_cast<uintptr_t>(&value)};
  dev.ioctl(DRM_IOCTL_AMDXDNA_GET_INFO, &arg);
  return value;
}

device::device(uint32_t device_index) : m_pdev(device_index) {
  auto metadata = get_info<amdxdna_drm_query_aie_metadata>(m_pdev, DRM_AMDXDNA_QUERY_AIE_METADATA);
  n_rows = metadata.core.row_count;
  n_cols = metadata.cols;
}

device::~device() = default;

const pdev& device::get_pdev() const { return m_pdev; }

std::unique_ptr<hw_ctx> device::create_hw_context(std::span<const uint8_t> pdi,
                                                  std::string_view cu_name) {
  return std::make_unique<hw_ctx>(*this, pdi, cu_name, n_rows, n_cols);
}

std::unique_ptr<bo> device::alloc_bo(size_t size, uint32_t flags) {
  return std::make_unique<bo>(m_pdev, size, flags);
}

std::unique_ptr<bo> device::alloc_user_bo(size_t size) {
  return std::make_unique<bo>(m_pdev, size, user_memory{});
}

clocks device::query_clocks() const {
  auto c = get_info<amdxdna_drm_query_clock_metadata>(m_pdev, DRM_AMDXDNA_QUERY_CLOCK_METADATA);
  return {c.mp_npu_clock.freq_mhz, c.h_clock.freq_mhz};
}

uint64_t device::busy_ns() const {
  // "drm-engine-<name>:\t<N> ns" lines (one engine on AIE2)
  std::ifstream f("/proc/self/fdinfo/" + std::to_string(m_pdev.m_dev_fd));
  uint64_t total = 0;
  for (std::string line; std::getline(f, line);) {
    if (!line.starts_with("drm-engine-")) continue;
    auto pos = line.find_first_of("0123456789", line.find(':'));
    if (pos == std::string::npos) continue;
    uint64_t ns = 0;
    std::from_chars(line.data() + pos, line.data() + line.size(), ns);
    total += ns;
  }
  return total;
}

std::optional<amdxdna_async_error> device::last_async_error() const {
  amdxdna_async_error err = {};
  amdxdna_drm_get_array ga = {};
  ga.param = DRM_AMDXDNA_HW_LAST_ASYNC_ERR;
  ga.element_size = sizeof(err);
  ga.num_element = 1;
  ga.buffer = reinterpret_cast<uintptr_t>(&err);
  m_pdev.ioctl(DRM_IOCTL_AMDXDNA_GET_ARRAY, &ga);
  if (!err.err_code && !err.ts_us) return std::nullopt;
  return err;
}

}  // namespace shim_xdna
