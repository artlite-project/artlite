#include "shim_xrt/xclbin.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <string_view>

#include "xrt/detail/xclbin.h"

namespace shim_xdna {

namespace {

// aiecc's values: the DPU kernel's id, its PDI's id and CDO group, the partition's QoS figures.
constexpr uint64_t kDpuKernelId = 0x901, kPdiId = 0x1;
constexpr uint32_t kPreCdoGroup = 0xC1, kOpsPerCycle = 2048;
constexpr uint64_t kInferenceFingerprint = 23423, kPrePostFingerprint = 12345;
constexpr uint16_t kColumns = 8;
constexpr uint32_t kBuffers = 5;  // bo0 .. bo4, the arguments from 3

// The kernel's arguments as xclbinutil writes them from aiecc's kernel JSON.
constexpr std::string_view kMetadata =
    R"(<?xml version="1.0" encoding="utf-8"?>
<project>
  <platform>
    <device>
      <core>
        <kernel name="MLIR_AIE" language="c" type="dpu">
          <extended-data subtype="1" functional="0" dpu_kernel_id="0x901"/>
          <arg name="opcode" addressQualifier="0" id="0" size="0x8" offset="0x00" hostOffset="0x0" hostSize="0x8" type="uint64_t"/>
          <arg name="instr" addressQualifier="1" id="1" size="0x8" offset="0x08" hostOffset="0x0" hostSize="0x8" type="char *"/>
          <arg name="ninstr" addressQualifier="0" id="2" size="0x4" offset="0x10" hostOffset="0x0" hostSize="0x4" type="uint32_t"/>
          <arg name="bo0" addressQualifier="1" id="3" size="0x8" offset="0x14" hostOffset="0x0" hostSize="0x8" type="void*"/>
          <arg name="bo1" addressQualifier="1" id="4" size="0x8" offset="0x1C" hostOffset="0x0" hostSize="0x8" type="void*"/>
          <arg name="bo2" addressQualifier="1" id="5" size="0x8" offset="0x24" hostOffset="0x0" hostSize="0x8" type="void*"/>
          <arg name="bo3" addressQualifier="1" id="6" size="0x8" offset="0x2C" hostOffset="0x0" hostSize="0x8" type="void*"/>
          <arg name="bo4" addressQualifier="1" id="7" size="0x8" offset="0x34" hostOffset="0x0" hostSize="0x8" type="void*"/>
          <instance name="MLIRAIE"/>
        </kernel>
      </core>
    </device>
  </platform>
</project>
)";

std::array<uint8_t, 32> sha256(std::span<const uint8_t> data) {
  static constexpr uint32_t K[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  auto rotr = [](uint32_t x, int n) { return x >> n | x << (32 - n); };
  auto block = [&](const uint8_t* p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
      w[i] = uint32_t(p[4 * i]) << 24 | uint32_t(p[4 * i + 1]) << 16 | uint32_t(p[4 * i + 2]) << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; i++)
      w[i] = w[i - 16] + (rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ w[i - 15] >> 3) + w[i - 7] +
             (rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ w[i - 2] >> 10);
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
    for (int i = 0; i < 64; i++) {
      const uint32_t t1 = k + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
      const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      k = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
    }
    h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += k;
  };
  const size_t full = data.size() / 64 * 64;
  for (size_t i = 0; i < full; i += 64) block(data.data() + i);
  uint8_t tail[128] = {};
  const size_t rest = data.size() - full, n = rest < 56 ? 64 : 128;
  std::memcpy(tail, data.data() + full, rest);
  tail[rest] = 0x80;
  for (int i = 0; i < 8; i++) tail[n - 1 - i] = static_cast<uint8_t>(uint64_t(data.size()) * 8 >> 8 * i);
  for (size_t i = 0; i < n; i += 64) block(tail + i);
  std::array<uint8_t, 32> out;
  for (int i = 0; i < 32; i++) out[i] = static_cast<uint8_t>(h[i / 4] >> (24 - 8 * (i % 4)));
  return out;
}

// A version 4 UUID of 16 bytes of a digest (Python's uuid.UUID(bytes=..., version=4)).
void uuid_of(const uint8_t* digest, void* uuid) {
  uint8_t u[16];
  std::memcpy(u, digest, 16);
  u[6] = (u[6] & 0x0F) | 0x40;
  u[8] = (u[8] & 0x3F) | 0x80;
  std::memcpy(uuid, u, 16);
}

// A section's bytes: each piece at an 8-byte boundary, children (strings, arrays) before the arrays that point to
// them, as xclbinutil lays them out.
struct Blob {
  std::vector<char> bytes;

  uint32_t put(const void* p, size_t n) {
    const auto at = static_cast<uint32_t>(bytes.size());
    bytes.insert(bytes.end(), static_cast<const char*>(p), static_cast<const char*>(p) + n);
    bytes.resize((bytes.size() + 7) / 8 * 8);
    return at;
  }
  uint32_t put(std::string_view s) {
    const uint32_t at = put(s.data(), s.size());
    bytes.resize(std::max<size_t>(bytes.size(), at + s.size() + 1));  // (its terminating zero)
    bytes.resize((bytes.size() + 7) / 8 * 8);
    return at;
  }
};

std::vector<char> partition_section(std::span<const uint8_t> pdi, const uint8_t* digest) {
  Blob s;
  s.bytes.resize(sizeof(aie_partition));  // (written last)
  const uint32_t name = s.put(std::string_view{}), commit = s.put(std::string_view{});
  const uint16_t first_column = 0;
  const uint32_t columns = s.put(&first_column, sizeof first_column);
  const uint32_t image = s.put(pdi.data(), pdi.size());
  const uint32_t group_name = s.put(std::string_view("DPU"));
  const uint32_t kernel_ids = s.put(&kDpuKernelId, sizeof kDpuKernelId);
  const uint32_t pre_groups = s.put(&kPreCdoGroup, sizeof kPreCdoGroup);
  cdo_group g{};
  g.mpo_name = group_name;
  g.cdo_type = CT_PRIMARY;
  g.pdi_id = kPdiId;
  g.dpu_kernel_ids = {1, kernel_ids};
  g.pre_cdo_groups = {1, pre_groups};
  const uint32_t groups = s.put(&g, sizeof g);
  aie_pdi p{};
  uuid_of(digest, &p.uuid);
  p.pdi_image = {static_cast<uint32_t>(pdi.size()), image};
  p.cdo_groups = {1, groups};
  const uint32_t pdis = s.put(&p, sizeof p);
  aie_partition h{};
  h.mpo_name = name;
  h.operations_per_cycle = kOpsPerCycle;
  h.inference_fingerprint = kInferenceFingerprint;
  h.pre_post_fingerprint = kPrePostFingerprint;
  h.info.column_width = kColumns;
  h.info.start_columns = {1, columns};
  h.aie_pdi = {1, pdis};
  h.kernel_commit_id = commit;
  std::memcpy(s.bytes.data(), &h, sizeof h);
  return std::move(s.bytes);
}

// The memory banks: control code in SRAM, the buffers in HOST.
std::vector<char> topology_section() {
  mem_data banks[2] = {};
  const char* tags[2] = {"HOST", "SRAM"};
  const uint64_t kb[2] = {0x10000, 0xC000};
  for (int i = 0; i < 2; i++) {
    banks[i].m_type = MEM_DRAM;
    banks[i].m_used = 1;
    banks[i].m_size = kb[i];
    banks[i].m_base_address = 0x4000000;
    std::memcpy(banks[i].m_tag, tags[i], std::strlen(tags[i]));
  }
  std::vector<char> out(offsetof(mem_topology, m_mem_data) + sizeof banks);
  const int32_t count = 2;
  std::memcpy(out.data(), &count, sizeof count);
  std::memcpy(out.data() + offsetof(mem_topology, m_mem_data), banks, sizeof banks);
  return out;
}

std::vector<char> ip_layout_section() {
  ip_data ip{};
  ip.m_type = IP_PS_KERNEL;
  ip.ps_kernel.m_subtype = ST_DPU;
  ip.ps_kernel.m_functional = FC_DPU;
  ip.ps_kernel.m_kernel_id = static_cast<uint16_t>(kDpuKernelId);
  ip.m_base_address = ~uint64_t(0);  // (not used)
  const std::string_view name = "MLIR_AIE:MLIRAIE";
  std::memcpy(ip.m_name, name.data(), name.size());
  std::vector<char> out(offsetof(ip_layout, m_ip_data) + sizeof ip);
  const int32_t count = 1;
  std::memcpy(out.data(), &count, sizeof count);
  std::memcpy(out.data() + offsetof(ip_layout, m_ip_data), &ip, sizeof ip);
  return out;
}

// The arguments' banks (the control code's, then the buffers'), `copies` times (GROUP_CONNECTIVITY: twice).
std::vector<char> connectivity_section(int copies) {
  std::vector<connection> c;
  for (int k = 0; k < copies; k++) {
    c.push_back({1, 0, 1});
    for (uint32_t i = 0; i < kBuffers; i++) c.push_back({static_cast<int32_t>(3 + i), 0, 0});
  }
  std::vector<char> out(offsetof(connectivity, m_connection) + c.size() * sizeof(connection));
  const auto count = static_cast<int32_t>(c.size());
  std::memcpy(out.data(), &count, sizeof count);
  std::memcpy(out.data() + offsetof(connectivity, m_connection), c.data(), c.size() * sizeof(connection));
  return out;
}

}  // namespace

std::vector<char> pdi_xclbin(std::span<const uint8_t> pdi) {
  const auto digest = sha256(pdi);
  struct Section {
    axlf_section_kind kind;
    std::vector<char> data;
  };
  const Section sections[] = {
      {MEM_TOPOLOGY, topology_section()},
      {AIE_PARTITION, partition_section(pdi, digest.data())},
      {EMBEDDED_METADATA, {kMetadata.begin(), kMetadata.end()}},
      {IP_LAYOUT, ip_layout_section()},
      {CONNECTIVITY, connectivity_section(1)},
      {ASK_GROUP_CONNECTIVITY, connectivity_section(2)},
      {ASK_GROUP_TOPOLOGY, topology_section()},
  };
  constexpr size_t n = std::size(sections);
  size_t at = offsetof(axlf, m_sections) + n * sizeof(axlf_section_header);
  axlf_section_header headers[n] = {};
  for (size_t i = 0; i < n; i++) {
    headers[i].m_sectionKind = sections[i].kind;
    headers[i].m_sectionOffset = at;
    headers[i].m_sectionSize = sections[i].data.size();
    at = (at + sections[i].data.size() + 7) / 8 * 8;
  }
  const size_t length = headers[n - 1].m_sectionOffset + headers[n - 1].m_sectionSize;

  axlf top{};
  std::memcpy(top.m_magic, "xclbin2", 8);
  top.m_signature_length = -1;
  std::memset(top.reserved, 0xFF, sizeof top.reserved);
  std::memset(top.m_keyBlock, 0xFF, sizeof top.m_keyBlock);
  std::memcpy(&top.m_uniqueId, digest.data() + 16, sizeof top.m_uniqueId);
  top.m_header.m_length = length;
  top.m_header.m_mode = XCLBIN_FLAT;
  uuid_of(digest.data() + 16, &top.m_header.uuid);
  top.m_header.m_numSections = n;

  std::vector<char> out(length);
  std::memcpy(out.data(), &top, offsetof(axlf, m_sections));
  std::memcpy(out.data() + offsetof(axlf, m_sections), headers, sizeof headers);
  for (size_t i = 0; i < n; i++)
    std::memcpy(out.data() + headers[i].m_sectionOffset, sections[i].data.data(), sections[i].data.size());
  return out;
}

}  // namespace shim_xdna
