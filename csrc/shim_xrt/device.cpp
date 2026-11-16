#include "shim_xrt/device.h"

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>

#include "xrt/xrt_device.h"

namespace shim_xdna {

namespace {
std::mutex g_mutex;
std::vector<const hw_ctx*> g_contexts;
std::weak_ptr<region> g_shared;  // (freed with its last buffer)
std::vector<std::weak_ptr<region>> g_regions;  // (every region made; the freed ones leave at the next one)
std::atomic<uint64_t> g_regions_gone{0};

constexpr size_t kPage = 4096;
size_t pages(size_t n) { return (std::max<size_t>(n, 1) + kPage - 1) / kPage * kPage; }

// (g_mutex held)
void note_region(const std::shared_ptr<region>& r) {
  std::erase_if(g_regions, [](const std::weak_ptr<region>& w) { return w.expired(); });
  g_regions.push_back(r);
}

// A region of its own (its memory populated here: without g_mutex).
std::shared_ptr<region> make_region(size_t bytes) {
  auto r = std::make_shared<region>(bytes);
  std::lock_guard lock(g_mutex);
  note_region(r);
  return r;
}
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

// One XRT device per index, as the amdxdna shim's pdev: a region's XRT buffer belongs to the device object it was made
// on, and every context's commands list it (on Windows a buffer of another xrt::device times the command out).
device::device(uint32_t device_index) {
  static std::mutex lock;
  static std::map<uint32_t, std::weak_ptr<xrt::device>> devices;
  std::lock_guard guard(lock);
  std::weak_ptr<xrt::device>& w = devices[device_index];
  m_dev = w.lock();
  if (!m_dev) w = m_dev = std::make_shared<xrt::device>(device_index);
}

device::~device() = default;

std::shared_ptr<region> device::take(size_t bytes, size_t& offset) {
  offset = 0;
  if (bytes >= kSharedMax) return make_region(bytes);
  std::lock_guard lock(g_mutex);
  std::shared_ptr<region> shared = g_shared.lock();
  if (!shared || shared->used + bytes > shared->size) {
    shared = std::make_shared<region>(kSharedBytes);
    note_region(shared);
    g_shared = shared;
  }
  offset = shared->used;
  shared->used += bytes;
  return shared;
}

std::vector<std::unique_ptr<bo>> device::alloc_bos(std::span<const size_t> sizes, size_t) {
  size_t total = 0;
  for (size_t n : sizes) total += pages(n);
  std::vector<std::unique_ptr<bo>> out;
  if (sizes.empty()) return out;
  std::shared_ptr<region> r = make_region(total);
  size_t offset = 0;
  for (size_t n : sizes) {
    out.push_back(std::make_unique<bo>(*this, n, r, offset));
    offset += pages(n);
  }
  return out;
}

std::vector<std::shared_ptr<region>> device::regions() {
  std::vector<std::shared_ptr<region>> out;
  {
    std::lock_guard lock(g_mutex);
    for (const std::weak_ptr<region>& w : g_regions)
      if (auto r = w.lock()) out.push_back(std::move(r));
  }
  std::erase_if(out, [](const std::shared_ptr<region>& r) { return !r->made(); });
  std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a->size > b->size; });
  return out;
}

void device::region_gone() { g_regions_gone++; }

uint64_t device::regions_gone() { return g_regions_gone; }

std::unique_ptr<bo> device::alloc_bo(size_t size, uint32_t flags) {
  return std::make_unique<bo>(*this, size, (flags & XCL_BO_FLAGS_CACHEABLE) ? bo::kind::instructions : bo::kind::data);
}

std::unique_ptr<bo> device::alloc_user_bo(size_t size) { return std::make_unique<bo>(*this, size, user_memory{}); }

std::unique_ptr<hw_ctx> device::create_hw_context(std::span<const uint8_t> pdi, std::string_view kernel_name) {
  return std::make_unique<hw_ctx>(*this, pdi, kernel_name);
}

clocks device::query_clocks() const { return {0, 0}; }

uint64_t device::busy_ns() const { return 0; }

std::optional<amdxdna_async_error> device::last_async_error() const { return std::nullopt; }

}  // namespace shim_xdna
