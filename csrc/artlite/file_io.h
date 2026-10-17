#pragma once

#include <cstddef>
#include <cstdint>
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

}  // namespace artlite
