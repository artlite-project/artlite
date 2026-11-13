// ParameterScratchpad (artlite/scratchpad.h): the parameters file's parser and the host-side range checks. Host only.
#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "artlite/scratchpad.h"

namespace {

// The parameters file's text in a temporary file removed with the object.
struct ParamsFile {
  std::filesystem::path path;
  explicit ParamsFile(const std::string& text) {
    static int count = 0;
    path = std::filesystem::temp_directory_path() /
           ("artlite_params_" + std::to_string(getpid()) + "_" + std::to_string(count++) + ".txt");
    std::ofstream(path) << text;
  }
  ~ParamsFile() { std::filesystem::remove(path); }
};

// <num_parameters> <num_slots>, then per parameter: name slot type kind min max; the joint bounds; the commands
const char* kBasic =
    "5 4\n"
    "core_p 0 i32 core - -\n"
    "addr_p 1 i32 addr 0 10\n"
    "repeat_p 2 i32 repeat - -\n"
    "off 3 i32 addr 0 100\n"
    "len - i32 addr 0 100\n"
    "0\n"
    "0\n";

std::vector<uint32_t> filled(size_t words) { return std::vector<uint32_t>(words, 0xDEADBEEF); }

}  // namespace

TEST(Scratchpad, ClearsItsSlotsAndShiftsByKind) {
  ParamsFile f(kBasic);
  auto buf = filled(6);
  artlite::ParameterScratchpad sp(buf.data(), buf.size() * 4, f.path.string());
  for (int i = 0; i < 4; i++) EXPECT_EQ(buf[i], 0u) << i;
  EXPECT_EQ(buf[4], 0xDEADBEEFu);  // (past the file's 4 slots)

  sp.write<uint32_t>("core_p", 5);  // (core: << 2)
  sp.write<uint32_t>("repeat_p", 3);  // (repeat: << 16)
  sp.write<uint32_t>("addr_p", 7);  // (addr: as it is)
  EXPECT_EQ(buf[0], 5u << 2);
  EXPECT_EQ(buf[2], 3u << 16);
  EXPECT_EQ(buf[1], 7u);
}

TEST(Scratchpad, WritesTypedValuesByTheirBits) {
  ParamsFile f(kBasic);
  auto buf = filled(4);
  artlite::ParameterScratchpad sp(buf.data(), buf.size() * 4, f.path.string());
  sp.write<uint16_t>("core_p", 0xFFFF);
  EXPECT_EQ(buf[0], 0xFFFFu << 2);
  sp.write<float>("repeat_p", 1.0f);
  EXPECT_EQ(buf[2], 0x3F800000u << 16);
}

TEST(Scratchpad, RefusesAValueOutsideItsRange) {
  ParamsFile f(kBasic);
  auto buf = filled(4);
  artlite::ParameterScratchpad sp(buf.data(), buf.size() * 4, f.path.string());
  EXPECT_THROW(sp.write<uint32_t>("addr_p", 11), std::invalid_argument);
  EXPECT_THROW(sp.write<int32_t>("addr_p", -1), std::invalid_argument);
  EXPECT_EQ(buf[1], 0u);  // (nothing written)
  sp.write<uint32_t>("addr_p", 10);
  EXPECT_EQ(buf[1], 10u);
}

TEST(Scratchpad, RefusesUnknownAndUnslottedParameters) {
  ParamsFile f(kBasic);
  auto buf = filled(4);
  artlite::ParameterScratchpad sp(buf.data(), buf.size() * 4, f.path.string());
  EXPECT_THROW(sp.write<uint32_t>("nope", 1), std::runtime_error);
  EXPECT_THROW(sp.write<uint32_t>("len", 1), std::runtime_error);  // (no command reads it: no slot)
}

TEST(Scratchpad, RefusesABufferSmallerThanItsSlots) {
  ParamsFile f(kBasic);
  auto buf = filled(3);
  EXPECT_THROW(artlite::ParameterScratchpad(buf.data(), buf.size() * 4, f.path.string()), std::runtime_error);
}

TEST(Scratchpad, ChecksJointBoundsAtValidate) {
  // offset + 4 * length <= 100
  ParamsFile f(
      "2 2\n"
      "off 0 i32 addr 0 100\n"
      "len 1 i32 addr 0 100\n"
      "1\n"
      "len off 4 100\n"
      "0\n");
  auto buf = filled(2);
  artlite::ParameterScratchpad sp(buf.data(), buf.size() * 4, f.path.string());
  sp.validate();  // (both 0)
  sp.write<uint32_t>("off", 40);
  sp.write<uint32_t>("len", 15);  // 40 + 60 = 100
  sp.validate();
  sp.write<uint32_t>("len", 16);  // 40 + 64
  EXPECT_THROW(sp.validate(), std::invalid_argument);
  sp.write<uint32_t>("off", 36);  // 36 + 64 = 100
  sp.validate();
}

TEST(Scratchpad, RefusesMalformedFiles) {
  auto buf = filled(8);
  auto load = [&](const std::string& text) {
    ParamsFile f(text);
    artlite::ParameterScratchpad sp(buf.data(), buf.size() * 4, f.path.string());
  };
  EXPECT_NO_THROW(load("1 1\na 0 i32 core - -\n0\n0\n"));
  EXPECT_THROW(load(""), std::runtime_error);                                        // (no counts)
  EXPECT_THROW(load("1 1\na 0 i32 core\n"), std::runtime_error);                    // (a short entry)
  EXPECT_THROW(load("2 2\na 0 i32 core - -\na 1 i32 core - -\n0\n0\n"), std::runtime_error);  // (a name twice)
  EXPECT_THROW(load("1 1\na 1 i32 core - -\n0\n0\n"), std::runtime_error);           // (a slot past the file's)
  EXPECT_THROW(load("1 1\na x i32 core - -\n0\n0\n"), std::runtime_error);           // (a slot not a number)
  EXPECT_THROW(load("1 1\na 0 i32 other - -\n0\n0\n"), std::runtime_error);          // (a kind)
  EXPECT_THROW(load("1 1\na 0 i32 addr 0 x\n0\n0\n"), std::runtime_error);           // (a range)
  EXPECT_THROW(load("1 1\na 0 i32 core - -\n"), std::runtime_error);                 // (no joint bound count)
  EXPECT_THROW(load("1 1\na 0 i32 core - -\n1\na a 1 10\n0\n"), std::runtime_error);  // (a bound of an unranged one)
  EXPECT_THROW(load("1 1\na 0 i32 core - -\n0\n"), std::runtime_error);              // (no command count)
  EXPECT_THROW(load("1 1\na 0 i32 core - -\n0\n2\n"), std::runtime_error);           // (commands may share slots)
  EXPECT_THROW(artlite::ParameterScratchpad(buf.data(), 32, "/nonexistent/params.txt"), std::runtime_error);
}
