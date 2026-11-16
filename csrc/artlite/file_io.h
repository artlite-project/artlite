#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace shim_xdna {
struct bo;
}  // namespace shim_xdna

namespace artlite {

/// Copy nbytes of a file, from file_offset, into buf at offset. Up to 16 threads each pread a
/// 512 KB chunk into a cache-resident bounce buffer and stream it out with non-temporal AVX-512
/// stores, so the data goes to DRAM without the read-for-ownership of normal stores and without
/// filling the caches (a page-cached 2 GB weight image: 28 ms, read() 122 ms). Sync the range to
/// the device before a dispatch reads it, as after any host write.
void read_file(shim_xdna::bo& buf, size_t offset, const std::string& path, uint64_t file_offset,
               size_t nbytes);

/// A file's bytes. Throws Error for one that cannot be read.
uint64_t file_size(const std::string& path);

/// What the system should do with pages of a file mapping (page_hint): read them ahead into the file cache at once
/// (reads of scattered rows would fault them in from the disk one at a time), give the whole pages inside a range back
/// (they leave the process, stay in the file cache and come back on the next access), map cached ones into the process
/// ahead of its reads (Linux).
enum class PageHint { will_need, dont_need, populate };

/// The hint for the ranges [offset, offset + nbytes) of the mapping [map, map + size): Linux's madvise; on Windows
/// PrefetchVirtualMemory and, for dont_need, VirtualUnlock (which takes unlocked pages out of the working set).
/// Throws Error for a range outside the mapping.
void page_hint(const uint8_t* map, size_t size, PageHint hint, std::span<const int64_t> offsets, size_t nbytes);

}  // namespace artlite
