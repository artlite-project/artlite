#include "shim_xrt/device.h"

#include <algorithm>
#include <mutex>

#include "xrt/xrt_device.h"

namespace shim_xdna {

namespace {
std::mutex g_mutex;
std::vector<const hw_ctx*> g_contexts;
std::weak_ptr<region> g_shared;  // (freed with its last buffer)
}  // namespace

void device::add_context(const hw_ctx* ctx) {
  std::lock_guard lock(g_mutex);
  g_contexts.push_back(ctx);
}

void device::remove_context(const hw_ctx* ctx) {
  std::lock_guard lock(g_mutex);
  std::erase(g_contexts, ctx);
}

const hw_ctx* device::any_context(const device* preferred) {
  std::lock_guard lock(g_mutex);
  for (const hw_ctx* c : g_contexts)
    if (&c->m_device == preferred) return c;
  return g_contexts.empty() ? nullptr : g_contexts.back();
}

device::device(uint32_t device_index) : m_dev(std::make_unique<xrt::device>(device_index)) {}

device::~device() = default;

std::shared_ptr<region> device::take(size_t bytes, size_t& offset) {
  offset = 0;
  if (bytes >= kSharedMax) return std::make_shared<region>(bytes);
  std::lock_guard lock(g_mutex);
  std::shared_ptr<region> shared = g_shared.lock();
  if (!shared || shared->used + bytes > shared->size) {
    shared = std::make_shared<region>(kSharedBytes);
    g_shared = shared;
  }
  offset = shared->used;
  shared->used += bytes;
  return shared;
}

std::unique_ptr<bo> device::alloc_bo(size_t size, uint32_t flags) {
  return std::make_unique<bo>(*this, size, (flags & XCL_BO_FLAGS_CACHEABLE) ? bo::kind::instructions : bo::kind::data);
}

std::unique_ptr<bo> device::alloc_user_bo(size_t size) { return std::make_unique<bo>(*this, size, user_memory{}); }

std::unique_ptr<hw_ctx> device::create_hw_context(std::span<const uint8_t> xclbin, std::string_view kernel_name) {
  return std::make_unique<hw_ctx>(*this, xclbin, kernel_name);
}

clocks device::query_clocks() const { return {0, 0}; }

uint64_t device::busy_ns() const { return 0; }

std::optional<amdxdna_async_error> device::last_async_error() const { return std::nullopt; }

}  // namespace shim_xdna
