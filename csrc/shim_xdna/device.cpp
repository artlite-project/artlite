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
#include <map>
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

// The device heap: heap BOs of 64 MB (each within one 64 MB page of the NPU's addresses) up to
// the driver's maximum.
constexpr size_t heap_bo_size = size_t(64) << 20;
constexpr size_t heap_max_size = size_t(512) << 20;
}  // namespace

namespace shim_xdna {

pdev::pdev(uint32_t device_index) {
  std::string path = "/dev/accel/accel" + std::to_string(device_index);
  m_dev_fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
  if (m_dev_fd < 0) shim_err(errno, "Failed to open device {}", path);
  try {
    m_heap_range_size = heap_max_size + heap_bo_size;  // (room to align it)
    m_heap_range = ::mmap(nullptr, m_heap_range_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (m_heap_range == MAP_FAILED) {
      m_heap_range = nullptr;
      shim_err(errno, "mmap(len={}) failed", m_heap_range_size);
    }
    m_heap_cpu = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(m_heap_range) + heap_bo_size - 1) &
                                         ~(heap_bo_size - 1));
    add_heap_bo();
    m_heap_xdna_addr = m_heap_bos[0]->m_xdna_addr;
  } catch (...) {
    m_heap_bos.clear();
    if (m_heap_range) ::munmap(m_heap_range, m_heap_range_size);
    ::close(m_dev_fd);
    throw;
  }
}

// The heap BOs' handles first, then their mappings with the reservation (no hole in it meanwhile).
pdev::~pdev() {
  m_heap_bos.clear();
  ::munmap(m_heap_range, m_heap_range_size);
  ::close(m_dev_fd);
}

std::shared_ptr<pdev> pdev::open(uint32_t device_index) {
  static std::mutex lock;
  static std::map<uint32_t, std::weak_ptr<pdev>> pdevs;
  std::lock_guard guard(lock);
  std::weak_ptr<pdev>& w = pdevs[device_index];
  std::shared_ptr<pdev> p = w.lock();
  if (!p) w = p = std::make_shared<pdev>(device_index);
  return p;
}

void pdev::add_heap_bo() const {
  size_t n = m_heap_bos.size();
  if ((n + 1) * heap_bo_size > heap_max_size)
    shim_err(ENOSPC, "the device heap is at its maximum of {} MB", heap_max_size >> 20);
  m_heap_bos.push_back(std::make_unique<bo>(*this, heap_bo_size, heap_memory{m_heap_cpu + n * heap_bo_size}));
}

void pdev::create_dev_bo(amdxdna_drm_create_bo& cbo) const {
  std::lock_guard guard(m_heap_lock);
  // EAGAIN: the driver placed the BO past the heap BOs
  while (::ioctl(m_dev_fd, DRM_IOCTL_AMDXDNA_CREATE_BO, &cbo) == -1) {
    int err = errno;
    if (err != EAGAIN) shim_err(err, "DRM_IOCTL_AMDXDNA_CREATE_BO IOCTL failed");
    try {
      add_heap_bo();
    } catch (const std::exception& e) {
      shim_err(err, "DRM_IOCTL_AMDXDNA_CREATE_BO IOCTL failed (a DEV BO of {} bytes; the device heap of {} MB is full: {})",
               cbo.size, m_heap_bos.size() * (heap_bo_size >> 20), e.what());
    }
  }
}

void* pdev::dev_bo_address(uint64_t xdna_addr) const { return m_heap_cpu + (xdna_addr - m_heap_xdna_addr); }

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

device::device(uint32_t device_index) : m_pdev(pdev::open(device_index)) {
  auto metadata = get_info<amdxdna_drm_query_aie_metadata>(*m_pdev, DRM_AMDXDNA_QUERY_AIE_METADATA);
  n_rows = metadata.core.row_count;
  n_cols = metadata.cols;
}

device::~device() = default;

const pdev& device::get_pdev() const { return *m_pdev; }

std::unique_ptr<hw_ctx> device::create_hw_context(std::span<const uint8_t> pdi,
                                                  std::string_view cu_name) {
  return std::make_unique<hw_ctx>(*this, pdi, cu_name, n_rows, n_cols);
}

std::unique_ptr<bo> device::alloc_bo(size_t size, uint32_t flags) {
  return std::make_unique<bo>(*m_pdev, size, flags);
}

std::unique_ptr<bo> device::alloc_user_bo(size_t size) {
  return std::make_unique<bo>(*m_pdev, size, user_memory{});
}

clocks device::query_clocks() const {
  auto c = get_info<amdxdna_drm_query_clock_metadata>(*m_pdev, DRM_AMDXDNA_QUERY_CLOCK_METADATA);
  return {c.mp_npu_clock.freq_mhz, c.h_clock.freq_mhz};
}

uint64_t device::busy_ns() const {
  // "drm-engine-<name>:\t<N> ns" lines (one engine on AIE2)
  std::ifstream f("/proc/self/fdinfo/" + std::to_string(m_pdev->m_dev_fd));
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
  m_pdev->ioctl(DRM_IOCTL_AMDXDNA_GET_ARRAY, &ga);
  if (!err.err_code && !err.ts_us) return std::nullopt;
  return err;
}

}  // namespace shim_xdna
