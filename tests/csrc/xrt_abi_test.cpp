// The XRT backend's kernel arguments (shim_xrt/abi.h) against the metadata of the xclbin built around a PDI
// (shim_xrt/xclbin.cpp). Host only.
#include <gtest/gtest.h>

#include <cstdint>
#include <format>
#include <string>
#include <vector>

#include "shim_xrt/abi.h"
#include "shim_xrt/xclbin.h"

namespace {

std::string xclbin_text() {
  const std::vector<uint8_t> pdi(256, 7);
  const std::vector<char> x = shim_xdna::pdi_xclbin(pdi);
  return std::string(x.begin(), x.end());
}

}  // namespace

TEST(XrtAbi, MetadataDeclaresTheArgumentsOfTheAbi) {
  using namespace shim_xdna::xrt_abi;
  const std::string x = xclbin_text();
  auto has = [&](const std::string& s) { return x.find(s) != std::string::npos; };
  EXPECT_TRUE(has(std::format(R"(name="opcode" addressQualifier="0" id="{}")", kArgOpcode)));
  EXPECT_TRUE(has(std::format(R"(name="instr" addressQualifier="1" id="{}")", kArgInstr)));
  EXPECT_TRUE(has(std::format(R"(name="ninstr" addressQualifier="0" id="{}")", kArgWords)));
  for (size_t i = 0; i < kMaxBuffers; i++)
    EXPECT_TRUE(has(std::format(R"(name="bo{}" addressQualifier="1" id="{}")", i, kArgFirstBuffer + i))) << i;
  EXPECT_FALSE(has(std::format(R"(name="bo{}")", kMaxBuffers)));
  EXPECT_EQ(kArgCount, kArgFirstBuffer + kMaxBuffers);
}
