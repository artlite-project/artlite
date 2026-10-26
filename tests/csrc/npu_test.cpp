#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <span>
#include <string>

#include "artlite/executable.h"
#include "artlite/kernel.h"
#include "shim_xdna/bo.h"
#include "shim_xdna/device.h"
#include "shim_xdna/hwctx.h"
#include "shim_xdna/xrt_mem.h"

static std::string pdi_path(const char* name) {
  return std::string(TEST_DATA_DIR) + "/" + name + "/main.pdi";
}

static std::string insts_path(const char* name) {
  return std::string(TEST_DATA_DIR) + "/" + name + "/insts.bin";
}

struct NpuFixture {
  artlite::Executable exe;
  const artlite::KernelEntry* entry;
  shim_xdna::device dev{};
  std::unique_ptr<shim_xdna::hw_ctx> ctx;

  NpuFixture(const std::string& pdi, const std::string& insts)
      : exe(artlite::Executable::from_files(pdi, insts)),
        entry(&exe.entry("MLIR_AIE")),
        ctx(dev.create_hw_context(entry->pdi, "MLIR_AIE")) {}

  std::unique_ptr<shim_xdna::bo> make_instr_buf() {
    auto& insts = entry->asm_instr_runlist.at(0);
    auto buf = dev.alloc_bo(insts.size() * sizeof(uint32_t),
                            static_cast<uint32_t>(XRT_BO_FLAGS_CACHEABLE));
    std::memcpy(buf->map(), insts.data(), insts.size() * sizeof(uint32_t));
    buf->sync(shim_xdna::direction::host2device);
    return buf;
  }
};

TEST(NPU, Addone) {
  constexpr size_t N = 1024;
  NpuFixture npu(pdi_path("addone"), insts_path("addone"));
  auto instr_buf = npu.make_instr_buf();

  auto in = npu.dev.alloc_bo(N * sizeof(int32_t), static_cast<uint32_t>(XCL_BO_FLAGS_HOST_ONLY));
  auto out = npu.dev.alloc_bo(N * sizeof(int32_t), static_cast<uint32_t>(XCL_BO_FLAGS_HOST_ONLY));

  auto in_span = std::span{static_cast<int32_t*>(in->map()), N};
  std::iota(in_span.begin(), in_span.end(), 0);
  in->sync(shim_xdna::direction::host2device);
  std::memset(out->map(), 0, out->get_properties().size);
  out->sync(shim_xdna::direction::host2device);

  artlite::Kernel kern(npu.dev, *npu.ctx);
  kern.dispatch(*instr_buf, {std::ref(*in), std::ref(*out)});
  out->sync(shim_xdna::direction::device2host);

  auto o = std::span{static_cast<int32_t*>(out->map()), N};
  for (size_t i = 0; i < N; ++i) ASSERT_EQ(o[i], static_cast<int32_t>(i) + 1) << "index " << i;
}
