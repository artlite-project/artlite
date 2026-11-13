#pragma once

// The host memory operations both shims and the packers share: cache-line write-back for the device, fresh memory
// made ready for DMA, non-temporal copies.

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "artlite/platform.h"

namespace artlite::memory {

/// Write back and invalidate the cache lines of [p, p + len). CLFLUSH is not ordered against younger loads: without
/// the fences a read right after a device->host sync can still hit the stale line (the kernel's clflush_cache_range
/// fences the same way). CLFLUSHOPT is no faster on Zen 5.
void flush_lines(const void* p, size_t len);

/// The same for the 64-byte lines at the byte offsets ``offsets[0 .. count)`` of ``base``.
void flush_line_offsets(const void* base, const int64_t* offsets, size_t count);

/// Zeroed, fresh memory [mem, mem + n) (anonymous or just committed) faulted in and written back by several threads
/// (up to 8, a thread per 256 MB): no dirty line of the zeroing may later land over what the NPU wrote, and the
/// driver's pin would fault the pages in one thread (Linux, 16 GB: ~1.7 s; this: ~0.3 s). In pieces of 32 MB: a
/// populate holds mmap_lock, which other threads' mmap / brk wait for.
void populate_zeroed(void* mem, size_t n);

#ifndef _WIN32
/// Advice for memory the driver pins: 2 MB pages where the kernel has them (fewer page walks for the IOMMU and the
/// CPU), no copy of it in a forked child (pinned pages must not turn copy-on-write). The errno of a refused
/// MADV_DONTFORK, else 0.
int advise_pinned(void* mem, size_t n);
#endif

/// dst gets the n bytes of src through non-temporal stores where dst is 64-byte aligned (the CPU never reads an image
/// back: no cache flush needed for the device after them); the head up to the alignment and the tail go through
/// memcpy. No fence: the caller issues ``_mm_sfence`` after its last copy, before the device may read dst.
ARTLITE_AVX512_FN inline void stream_copy(void* dst_, const void* src_, size_t n) {
  auto* dst = static_cast<uint8_t*>(dst_);
  auto* src = static_cast<const uint8_t*>(src_);
  const size_t head = std::min(n, (64 - reinterpret_cast<uintptr_t>(dst) % 64) % 64);
  std::memcpy(dst, src, head);
  size_t i = head;
  for (; i + 256 <= n; i += 256) {
    __m512i a = _mm512_loadu_si512(src + i), b = _mm512_loadu_si512(src + i + 64);
    __m512i c = _mm512_loadu_si512(src + i + 128), d = _mm512_loadu_si512(src + i + 192);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i), a);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i + 64), b);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i + 128), c);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i + 192), d);
  }
  for (; i + 64 <= n; i += 64)
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i), _mm512_loadu_si512(src + i));
  std::memcpy(dst + i, src + i, n - i);
}

}  // namespace artlite::memory
