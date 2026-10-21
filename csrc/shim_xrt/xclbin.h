#ifndef SHIM_XRT_XCLBIN_H
#define SHIM_XRT_XCLBIN_H

#include <cstdint>
#include <span>
#include <vector>

namespace shim_xdna {

// The kernel of every xclbin of a PDI: opcode, control code, its length in words, five buffers (bo0 .. bo4).
inline constexpr const char* kXclbinKernel = "MLIR_AIE";

// The xclbin XRT opens a hardware context from, built around a PDI in memory: aiecc's sections in xclbinutil's
// layout (without its JSON mirror), on a partition of the whole array (as the amdxdna shim's contexts). Its UUIDs
// (the xclbin's, the PDI's) come from the PDI's SHA-256: equal PDIs give equal xclbins.
std::vector<char> pdi_xclbin(std::span<const uint8_t> pdi);

}  // namespace shim_xdna

#endif
