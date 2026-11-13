#pragma once

// The arguments of the XRT backend's kernel, which xclbin.cpp's metadata declares and kernel_xrt.cpp binds: the opcode,
// the instruction buffer, its length in words, then the data buffers bo0 .. bo<kMaxBuffers - 1>. The NPU reaches only
// the XRT buffers its command lists, so a kernel has no more.

#include <cstddef>

namespace shim_xdna::xrt_abi {

constexpr unsigned kArgOpcode = 0, kArgInstr = 1, kArgWords = 2, kArgFirstBuffer = 3;
constexpr size_t kMaxBuffers = 5;
constexpr unsigned kArgCount = kArgFirstBuffer + kMaxBuffers;  // (the arguments' ids are 0 .. kArgCount - 1)

static_assert(kArgOpcode == 0 && kArgInstr == 1 && kArgWords == 2 && kArgFirstBuffer == kArgWords + 1);

}  // namespace shim_xdna::xrt_abi
