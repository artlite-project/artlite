#ifndef SHIM_XRT_BO_H
#define SHIM_XRT_BO_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>

#include "shim_xdna/xrt_mem.h"

namespace xrt {
class bo;
class hw_context;
}

namespace shim_xdna {
struct device;
struct hw_ctx;

enum class direction {
  host2device,
  device2host,
};

struct properties {
  shim_xcl_bo_flags flags;
  uint64_t size;
  uint64_t paddr;
  uint64_t kmhdl;
};

// Tag of the user-memory constructor.
struct user_memory {};

// Host memory pinned as an XRT buffer per memory group (on Windows each context has its own groups), made against a
// context at its first use there (XRT creates a buffer only then). A data buffer is a range of a region: small ones
// share one, a large one has its own.
struct region {
  struct pinned {
    std::unique_ptr<xrt::hw_context> hold;  // (the context the XRT buffer was made against: it must outlive it)
    std::unique_ptr<xrt::bo> xrt;
  };

  explicit region(size_t size);
  ~region();
  region(const region&) = delete;
  region& operator=(const region&) = delete;

  xrt::bo& xrt_for(const hw_ctx& ctx, uint32_t group);

  void* host;
  size_t size;
  size_t used = 0;  // (shared regions: the bytes handed out)
  std::mutex lock;  // (by_group: dispatches of several threads)
  std::map<uint32_t, pinned> by_group;
};

// A buffer the NPU reads and writes. A data buffer is a range of a region (above); an instruction buffer is a
// device-side XRT buffer that `sync` fills from its host copy, because XRT refuses to build one from user memory.
struct bo {
  enum class kind { data, instructions };

  bo(device& dev, size_t size, kind k);
  bo(device& dev, size_t size, user_memory);
  ~bo();
  bo(const bo&) = delete;
  bo& operator=(const bo&) = delete;

  void* map() const { return m_host; }
  size_t size() const { return m_size; }
  // Both directions on Linux write back and invalidate the range's cache lines (the NPU's DMA does not snoop them).
  void sync(direction dir, size_t size, size_t offset);
  void sync(direction dir);
  properties get_properties() const;
  // The address the NPU's DMA uses for the first byte (XRT's, not the host's). It exists once the buffer has an XRT
  // handle, which needs a hardware context in the process: throws without one.
  uint64_t get_paddr();
  // The address in `ctx`: on Windows each context maps the buffer at its own address.
  uint64_t paddr_in(const hw_ctx& ctx);
  std::string_view type_to_name() const;

  // The XRT buffer for kernel argument `arg` of `ctx`'s kernel: exactly this buffer (a data buffer in a shared region:
  // a sub-buffer of it), created on first use.
  xrt::bo& xrt_for(const hw_ctx& ctx, unsigned arg);
  // The XRT buffer of a data buffer's whole region (what a command lists so that the NPU reaches the buffer).
  xrt::bo& region_for(const hw_ctx& ctx, unsigned arg);
  bool is_data() const { return m_kind == kind::data; }
  // Whether the transaction in this instruction buffer (its header's length) has a DDR patch: an address the firmware
  // takes from the command's k-th buffer. Walked once per upload.
  bool ddr_patched();

 private:
  void flush();  // (instruction buffers: the host copy's newer bytes to the XRT buffer)

  device& m_device;
  void* m_host = nullptr;
  size_t m_size = 0;
  kind m_kind;
  std::shared_ptr<region> m_region;  // (data buffers)
  size_t m_offset = 0;  // (in the region)
  std::mutex m_lock;  // (the XRT buffers below)
  uint32_t m_group = ~0u;
  std::unique_ptr<xrt::hw_context> m_hold;  // (instruction buffers: as region::hold)
  std::unique_ptr<xrt::bo> m_xrt;  // (instruction buffers: theirs)
  std::map<uint32_t, std::unique_ptr<xrt::bo>> m_subs;  // (data buffers in a shared region: their sub-buffer per group)
  size_t m_dirty_lo = ~size_t(0), m_dirty_hi = 0;  // (instruction buffers: the bytes where the host copy is newer)
  std::optional<bool> m_patched;  // (instruction buffers: ddr_patched since the last upload)
};

}  // namespace shim_xdna

#endif
