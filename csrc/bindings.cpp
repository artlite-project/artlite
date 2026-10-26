#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "artlite/dequant.h"
#include "artlite/error.h"
#include "artlite/executable.h"
#include "artlite/file_io.h"
#include "artlite/gguf_json.h"
#include "artlite/kernel.h"
#include "artlite/moe_host.h"
#include "artlite/profiler.h"
#include "artlite/shim.h"
#include "artlite/tile_pack.h"

namespace nb = nanobind;
using namespace nb::literals;

extern const char artlite_source_hash[];  // (source_hash.cpp, generated: CMakeLists.txt's ARTLITE_SOURCE_HASH)

class ParameterScratchpad {
 public:
  /// Construct from a raw buffer and its size in bytes (Python bindings /
  /// testing). The size is required: clear() and every write index off
  /// scratchpadSizeBytes, which parseParams derives from the params file, not
  /// from the buffer.
  ParameterScratchpad(uint32_t* buffer, size_t bufferSizeBytes, const std::string& paramsPath)
      : boMap(buffer) {
    parseParams(paramsPath);
    if (bufferSizeBytes < scratchpadSizeBytes)
      throw std::runtime_error("ParameterScratchpad: buffer size (" +
                               std::to_string(bufferSizeBytes) + ") < required scratchpad size (" +
                               std::to_string(scratchpadSizeBytes) + ")");
    clear();
  }

  /// Write a raw 32-bit value by name.  For core-kind parameters, the bits
  /// are left-shifted by 2 (firmware requirement); for repeat-kind ones, by
  /// 16 (into the task queue's repeat field).  For addr-kind parameters,
  /// the value is written directly (no shift). A DMA offset or length
  /// parameter's value, read as an int32, must lie in the range the compiler
  /// gave it, which keeps its transfers within their buffers.
  void writeBits(const std::string& name, uint32_t bits) {
    auto it = paramMap.find(name);
    if (it == paramMap.end()) {
      throw std::runtime_error("ParameterScratchpad: unknown parameter '" + name + "'");
    }
    if (auto range = ranges.find(name); range != ranges.end()) {
      int32_t value = static_cast<int32_t>(bits);
      if (value < range->second.first || value > range->second.second)
        throw std::invalid_argument("ParameterScratchpad: value " + std::to_string(value) + " of '" + name +
                                    "' is outside [" + std::to_string(range->second.first) + ", " +
                                    std::to_string(range->second.second) +
                                    "], which keeps its DMA transfers within their buffers");
      values[name] = value;
    }
    uint8_t idx = it->second;
    uint32_t encoded = bits;
    if (coreParams.count(name)) {
      // core parameters require shift-2 to survive masking of lowest bits by
      // firmware op
      encoded = bits << 2;
    } else if (repeatParams.count(name)) {
      encoded = bits << 16;
    }
    boMap[idx] = encoded;
  }

  /// Write a typed parameter value. For core-kind parameters, the raw bits
  /// are left-shifted by 2 as required by the firmware's UPDATE_REG Incr mode,
  /// and the core right-shifts (unsigned) by 2 after reading. The round trip
  /// therefore zeroes the top 2 bits of the value (the bottom 2 bits are
  /// preserved). Types that fit in 30 bits (e.g. uint16_t, int16_t,
  /// std::bfloat16_t, and uint32_t values < 2^30) round-trip losslessly.
  /// `float` is not supported as a core-kind parameter (the verifier in
  /// `--aie-lower-scratchpad-parameters` rejects it).
  template <typename T>
  void write(const std::string& name, T value) {
    static_assert(sizeof(T) <= 4, "Parameter values must be at most 32 bits");
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    writeBits(name, bits);
  }

  /// Check the written values of each offset and length parameter pair that
  /// one DMA transfer uses. Throws std::invalid_argument if they take the
  /// transfer past the end of its buffer.
  void validate() const {
    for (const JointBound& bound : jointBounds) {
      int64_t offset = values.at(bound.offset);
      int64_t length = values.at(bound.length);
      int64_t end = offset + bound.lengthStep * length;
      if (end > bound.max)
        throw std::invalid_argument("ParameterScratchpad: '" + bound.offset + "' = " + std::to_string(offset) +
                                    " plus " + std::to_string(bound.lengthStep) + " * '" + bound.length +
                                    "' = " + std::to_string(length) + " is " + std::to_string(end) + ", above " +
                                    std::to_string(bound.max) + ", so a DMA transfer using both runs past its buffer");
    }
  }

 private:
  uint32_t* boMap = nullptr;
  size_t scratchpadSizeBytes = 0;
  std::unordered_map<std::string, uint8_t> paramMap;
  std::unordered_set<std::string> coreParams;  // params with kind="core"
  std::unordered_set<std::string> repeatParams;  // params with kind="repeat"
  std::unordered_map<std::string, std::pair<int32_t, int32_t>> ranges;
  std::unordered_map<std::string, int32_t> values;  // of the ranged params
  struct JointBound {
    std::string length, offset;
    int64_t lengthStep, max;
  };
  std::vector<JointBound> jointBounds;

  void clear() {
    for (size_t i = 0; i < scratchpadSizeBytes / 4; i++) {
      boMap[i] = 0;
    }
  }

  void parseParams(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("ParameterScratchpad: cannot open '" + path + "'");
    }

    // Format (mlir-aie's emitScratchpadParamsFile):
    //   <num_parameters>
    //   <name> <state_table_idx> <type> <kind> <min> <max>
    //   ...
    //   <num_joint_bounds>
    //   <length> <offset> <length_step> <max>
    //   ...
    // where kind is "core", "addr" or "repeat", and <min> <max> is "- -" for a
    // parameter with no range. A joint bound requires
    // offset + length_step * length <= max.
    unsigned numParams = 0;
    file >> numParams;
    scratchpadSizeBytes = numParams * 4;

    for (unsigned i = 0; i < numParams; i++) {
      std::string name, type, kind, min, max;
      unsigned idx;
      file >> name >> idx >> type >> kind >> min >> max;
      if (!file)
        throw std::runtime_error("ParameterScratchpad: malformed entry " + std::to_string(i) + " in '" + path + "'");
      if (idx > 255)
        throw std::runtime_error("ParameterScratchpad: state_table_idx " + std::to_string(idx) +
                                 " for '" + name + "' exceeds uint8_t range");
      if (paramMap.count(name))
        throw std::runtime_error("ParameterScratchpad: duplicate parameter name '" + name + "'");
      paramMap[name] = static_cast<uint8_t>(idx);
      if (kind != "core" && kind != "addr" && kind != "repeat") {
        throw std::runtime_error("ParameterScratchpad: invalid kind '" + kind +
                                 "' for parameter '" + name + "'");
      } else if (kind == "core") {
        coreParams.insert(name);
      } else if (kind == "repeat") {
        repeatParams.insert(name);
      }
      if (min != "-") {
        try {
          ranges[name] = {std::stoi(min), std::stoi(max)};
        } catch (const std::logic_error&) {
          throw std::runtime_error("ParameterScratchpad: malformed range '" + min + " " + max + "' for parameter '" +
                                   name + "' in '" + path + "'");
        }
        values[name] = 0;
      }
    }

    unsigned numJointBounds = 0;
    file >> numJointBounds;
    if (!file) throw std::runtime_error("ParameterScratchpad: missing joint bound count in '" + path + "'");
    for (unsigned i = 0; i < numJointBounds; i++) {
      JointBound bound;
      file >> bound.length >> bound.offset >> bound.lengthStep >> bound.max;
      if (!file)
        throw std::runtime_error("ParameterScratchpad: malformed joint bound " + std::to_string(i) + " in '" + path +
                                 "'");
      for (const std::string& name : {bound.length, bound.offset})
        if (!ranges.count(name))
          throw std::runtime_error("ParameterScratchpad: joint bound " + std::to_string(i) + " names '" + name +
                                   "', which has no range, in '" + path + "'");
      jointBounds.push_back(bound);
    }
  }
};

static std::vector<std::reference_wrapper<shim_xdna::bo>> buffer_refs(nb::list data_bufs) {
  std::vector<std::reference_wrapper<shim_xdna::bo>> refs;
  refs.reserve(nb::len(data_bufs));
  for (nb::handle b : data_bufs) refs.emplace_back(nb::cast<shim_xdna::bo&>(b));
  return refs;
}

// Run a blocking dispatch or wait without the GIL; afterwards drop the buffer references of the
// async dispatches it retired (all of them unless a wait timed out).
template <typename F>
static void retire(artlite::Kernel& self, F&& f) {
  auto drop = [&] {
    if (!self.dispatch_poll()) return;
    nb::handle py_self = nb::find(self);
    if (nb::hasattr(py_self, "_inflight")) nb::setattr(py_self, "_inflight", nb::none());
  };
  try {
    nb::gil_scoped_release release;
    f();
  } catch (...) {
    drop();
    throw;
  }
  drop();
}

// NOLINTBEGIN(bugprone-unused-raii)
// The module's name: _native, or _native_<backend> for a second backend of the platform (artlite.backend).
#define ARTLITE_NB_MODULE(name, m) NB_MODULE(name, m)
ARTLITE_NB_MODULE(ARTLITE_MODULE, m) {
  m.doc() = "Artlite: lightweight NPU runtime";
#ifdef ARTLITE_SHIM_XRT
  m.attr("BACKEND") = "xrt";
  m.attr("XRT_DIR") = ARTLITE_XRT_DIR;  // (the XRT the module was built against: its tools)
#else
  m.attr("BACKEND") = "xdna";
#endif
  m.def(
      "source_hash", [] { return std::string(artlite_source_hash); },
      "The hash of the sources the module was built from (CMakeLists.txt's ARTLITE_SOURCE_HASH; "
      "artlite.backend.stale_native compares it with the checkout's).");

  // Zero-copy numpy/torch views of Buffer mappings (map()) keep their
  // Buffer/Device wrappers alive until after nanobind's exit-time leak audit,
  // producing false "leaked N instances" reports even though gc finds zero
  // live instances at teardown and the actual munmap/GEM-close runs in ~bo().
  // No memory is lost when the process exits. Suppress the spurious audit;
  // ARTLITE_NB_LEAK_WARNINGS=1 re-enables it when hunting real leaks.
  if (!std::getenv("ARTLITE_NB_LEAK_WARNINGS"))
    nb::set_leak_warnings(false);

  // --- Buffer (wraps shim_xdna::bo directly) ---
  nb::class_<shim_xdna::bo>(m, "Buffer")
      .def("__repr__",
           [](shim_xdna::bo& self) -> std::string {
             auto props = self.get_properties();
             return std::format("Buffer(type={}, size={} bytes, paddr=0x{:x})", self.type_to_name(),
                                props.size, props.paddr);
           })
      .def_prop_ro("size", [](shim_xdna::bo& self) -> size_t { return self.size(); })
      .def(
          "map",
          [](shim_xdna::bo& self) {
            void* ptr = self.map();
            size_t shape[1] = {self.size()};
            return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(ptr, 1, shape, nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def(
          "write",
          [](shim_xdna::bo& self, nb::ndarray<uint8_t, nb::ndim<1>> data, size_t offset) {
            size_t nbytes = data.shape(0);
            if (offset > self.size() || nbytes > self.size() - offset)
              throw artlite::Error("write: offset + nbytes exceeds buffer size");
            std::memcpy(static_cast<uint8_t*>(self.map()) + offset, data.data(), nbytes);
          },
          "data"_a, "offset"_a = 0)
      .def(
          "read_file",
          [](shim_xdna::bo& self, const std::string& path, size_t offset, uint64_t file_offset,
             int64_t nbytes) {
            if (nbytes < 0) {
              std::ifstream f(path, std::ios::binary | std::ios::ate);
              if (!f) throw artlite::Error("read_file: cannot open '" + path + "'");
              nbytes = static_cast<int64_t>(f.tellg()) - static_cast<int64_t>(file_offset);
            }
            nb::gil_scoped_release release;
            artlite::read_file(self, offset, path, file_offset, static_cast<size_t>(nbytes));
          },
          "path"_a, "offset"_a = 0, "file_offset"_a = 0, "nbytes"_a = -1,
          "Copy nbytes (default: the rest) of a file from file_offset into the buffer at offset: "
          "parallel reads, non-temporal stores (bypassing the caches). Sync before the device reads "
          "it.")
      .def("sync_to_device",
           [](shim_xdna::bo& self) { self.sync(shim_xdna::direction::host2device); })
      .def("sync_from_device",
           [](shim_xdna::bo& self) { self.sync(shim_xdna::direction::device2host); })
      .def(
          "sync_to_device",
          [](shim_xdna::bo& self, size_t size, size_t offset) {
            self.sync(shim_xdna::direction::host2device, size, offset);
          },
          "size"_a, "offset"_a)
      .def(
          "sync_from_device",
          [](shim_xdna::bo& self, size_t size, size_t offset) {
            self.sync(shim_xdna::direction::device2host, size, offset);
          },
          "size"_a, "offset"_a)
      .def_prop_ro("paddr", [](shim_xdna::bo& self) -> uint64_t { return self.get_paddr(); })
      .def(
          "paddr_in", [](shim_xdna::bo& self, const shim_xdna::hw_ctx& ctx) -> uint64_t { return self.paddr_in(ctx); },
          "ctx"_a, "The device address in a hardware context (XRT on Windows maps a buffer at its own in each).");

  // --- HwContext (wraps shim_xdna::hw_ctx directly) ---
  nb::class_<shim_xdna::hw_ctx>(m, "HwContext")
      .def("read_aie_reg", &shim_xdna::hw_ctx::read_aie_reg, "col"_a, "row"_a, "reg_addr"_a,
           nb::sig("def read_aie_reg(self, col: int, row: int, reg_addr: int) -> int"))
      .def(
          "status",
          [](const shim_xdna::hw_ctx& self) {
            auto e = self.status();
            nb::dict d;
            d["start_col"] = e.start_col;
            d["num_col"] = e.num_col;
            d["command_submissions"] = e.command_submissions;
            d["command_completions"] = e.command_completions;
            d["state"] = e.state;
            d["txn_op_idx"] = e.txn_op_idx;
            d["ctx_pc"] = e.ctx_pc;
            d["fatal_error_type"] = e.fatal_error_type;
            d["fatal_error_exception_type"] = e.fatal_error_exception_type;
            d["fatal_error_exception_pc"] = e.fatal_error_exception_pc;
            d["fatal_error_app_module"] = e.fatal_error_app_module;
            return d;
          },
          "The driver's entry of this context, with the firmware's health report (txn_op_idx, "
          "ctx_pc, fatal error fields). A firmware query: only after a clean failure, never while "
          "a job may be hung.",
          nb::sig("def status(self) -> dict[str, int]"));

  // --- KernelEntry ---
  nb::class_<artlite::KernelEntry>(m, "KernelEntry")
      .def_prop_ro(
          "asm_instr_runlist",
          [](const artlite::KernelEntry& self) {
            nb::list result;
            for (auto& insts : self.asm_instr_runlist) {
              size_t shape[1] = {insts.size()};
              result.append(nb::ndarray<nb::numpy, const uint32_t, nb::ndim<1>>(
                  insts.data(), 1, shape, nb::handle()));
            }
            return result;
          },
          nb::rv_policy::reference_internal);

  // --- Executable ---
  nb::class_<artlite::Executable>(m, "Executable")
      .def_static("from_files", &artlite::Executable::from_files, "pdi_path"_a, "insts_path"_a)
      .def(
          "entry",
          [](const artlite::Executable& self) -> const artlite::KernelEntry& {
            return self.entry(0);
          },
          nb::rv_policy::reference_internal);

  // --- Device (wraps shim_xdna::device directly) ---
  nb::class_<shim_xdna::device>(m, "Device")
      .def(nb::init<uint32_t>(), "device_index"_a = 0)
      .def(
          "alloc_host_buffer",
          [](shim_xdna::device& self, size_t size) {
            return self.alloc_bo(size, static_cast<uint32_t>(XCL_BO_FLAGS_HOST_ONLY));
          },
          "size"_a, nb::keep_alive<0, 1>())
      .def(
          "alloc_user_buffer",
          [](shim_xdna::device& self, size_t size) {
            nb::gil_scoped_release release;
            return self.alloc_user_bo(size);
          },
          "size"_a, nb::keep_alive<0, 1>(),
          "A zeroed host buffer in this process's memory (2 MB pages faulted in by several threads, pinned by the "
          "driver): for large buffers (16 GB: ~0.3 s, alloc_host_buffer 1.7 s). Like alloc_host_buffer otherwise.")
      .def(
          "alloc_insts_buffer",
          [](shim_xdna::device& self, size_t size) {
            return self.alloc_bo(size, static_cast<uint32_t>(XRT_BO_FLAGS_CACHEABLE));
          },
          "size"_a, nb::keep_alive<0, 1>())
      // create_context from Executable (uses first entry)
      // Each context uses a host resident 64 MB buffer which is memory mapped into
      // the ERT instance created to service the workload. The ctrlcode used by the
      // workload is copied into this special memory.
      .def(
          "create_context",
          [](shim_xdna::device& self, const artlite::Executable& exe) {
            auto& e = exe.entry(0);
            return self.create_hw_context(e.pdi, e.name);
          },
          "exe"_a, nb::keep_alive<0, 1>())
      // create_context from raw PDI + kernel name
      .def(
          "create_context",
          [](shim_xdna::device& self, nb::ndarray<nb::numpy, const uint8_t, nb::ndim<1>> pdi,
             std::string_view kernel_name) {
            std::span<const uint8_t> sp(pdi.data(), pdi.shape(0));
            return self.create_hw_context(sp, kernel_name);
          },
          "pdi"_a, "kernel_name"_a, nb::keep_alive<0, 1>())
      .def(
          "clocks",
          [](const shim_xdna::device& self) {
            auto c = self.query_clocks();
            return std::make_pair(c.mp_npu_mhz, c.h_mhz);
          },
          "(MP-NPU clock, H clock) in MHz as the driver reports them.",
          nb::sig("def clocks(self) -> tuple[int, int]"))
      .def("driver_version", &shim_xdna::device::driver_version,
           "The DRM driver's name and version: amdxdna_accel_driver 0.6 / 0.8 / 0.10 are Linux 7.0 / 7.1 / 7.2's "
           "own (\"\" on XRT).")
      .def("power_mode", &shim_xdna::device::power_mode,
           "The NPU's power mode: 0 default, 1 low, 2 medium, 3 high, 4 turbo (-1 on XRT: xrt-smi reports it).")
      .def("heap_free", &shim_xdna::device::heap_free,
           "Bytes of the process's device heap (DEV BOs: alloc_insts_buffer) not in use yet, as the driver spends it "
           "(each BO rounded up to 32 KB, 128 KB per hardware context), of the heap BOs mapped so far.")
      .def_prop_ro("heap_fixed", &shim_xdna::device::heap_fixed,
                   "The driver refused to grow the device heap: one 64 MB heap per process (Linux before 7.2).")
      .def("grow_heap", &shim_xdna::device::grow_heap,
           "Add a 64 MB heap BO to the device heap; False where the driver cannot grow it (then heap_fixed) or at its "
           "maximum.")
      .def("busy_ns", &shim_xdna::device::busy_ns,
           "Device busy time of this Device's commands so far (fdinfo drm-engine, mailbox send to "
           "response). Reads a /proc file: keep it out of timed regions.")
      .def(
          "last_async_error",
          [](const shim_xdna::device& self) -> nb::object {
            auto e = self.last_async_error();
            if (!e) return nb::none();
            nb::dict d;
            d["err_code"] = e->err_code;
            d["ts_us"] = e->ts_us;
            d["ex_err_code"] = e->ex_err_code;
            return d;
          },
          "The most recent asynchronous AIE error (device-wide), or None. A firmware query: only "
          "after a clean failure, never while a job may be hung.",
          nb::sig("def last_async_error(self) -> dict[str, int] | None"));

  // --- Kernel ---
  // A pending asynchronous dispatch keeps its buffers referenced (in _inflight) until a wait
  // retires it: the driver does not keep them alive.
  nb::class_<artlite::Kernel>(m, "Kernel", nb::dynamic_attr())
      .def(nb::init<shim_xdna::device&, shim_xdna::hw_ctx&>(), "dev"_a, "ctx"_a,
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
      .def(nb::init<shim_xdna::device&, shim_xdna::hw_ctx&, std::string_view>(), "dev"_a, "ctx"_a,
           "kernel_name"_a, nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
      .def(
          "dispatch",
          [](artlite::Kernel& self, shim_xdna::bo& ctrl_code, nb::list data_bufs,
             uint32_t timeout_ms, int64_t spin_after_us, int spin_cpu) {
            auto refs = buffer_refs(data_bufs);
            retire(self, [&] { self.dispatch(ctrl_code, refs, timeout_ms, spin_after_us, spin_cpu); });
          },
          "ctrl_code"_a, "data_bufs"_a, "timeout_ms"_a = 0, "spin_after_us"_a = -1, "spin_cpu"_a = 0,
          "Run a transaction and wait for it. spin_after_us >= 0: sleep that long after the submit, then poll for "
          "the end on CPU spin_cpu (the one taking the NPU's interrupts: awake when the completion comes).",
          nb::sig("def dispatch(self, ctrl_code: Buffer, data_bufs: list[Buffer], timeout_ms: int "
                  "= 0, spin_after_us: int = -1, spin_cpu: int = 0) -> None"))
      .def(
          "dispatch_elf",
          [](artlite::Kernel& self, shim_xdna::bo& instr_bo, nb::list data_bufs,
             uint32_t timeout_ms, size_t instr_bytes) {
            auto refs = buffer_refs(data_bufs);
            retire(self, [&] { self.dispatch_elf(instr_bo, refs, timeout_ms, instr_bytes); });
          },
          "instr_bo"_a, "data_bufs"_a, "timeout_ms"_a = 0, "instr_bytes"_a = 0,
          nb::sig("def dispatch_elf(self, instr_bo: Buffer, data_bufs: list[Buffer], timeout_ms: "
                  "int = 0, instr_bytes: int = 0) -> None"))
      // --- Async dispatch ---
      .def(
          "dispatch_elf_async",
          [](artlite::Kernel& self, nb::handle instr_bo, nb::list data_bufs, size_t instr_bytes) {
            auto refs = buffer_refs(data_bufs);
            self.dispatch_elf_async(nb::cast<shim_xdna::bo&>(instr_bo), refs, instr_bytes);
            nb::handle py_self = nb::find(self);
            nb::object inflight = nb::getattr(py_self, "_inflight", nb::none());
            if (inflight.is_none()) {
              inflight = nb::list();
              nb::setattr(py_self, "_inflight", inflight);
            }
            nb::list l = nb::borrow<nb::list>(inflight);
            l.append(instr_bo);
            l.append(data_bufs);
          },
          "instr_bo"_a, "data_bufs"_a, "instr_bytes"_a = 0,
          nb::sig("def dispatch_elf_async(self, instr_bo: Buffer, data_bufs: list[Buffer], "
                  "instr_bytes: int = 0) -> None"))
      .def(
          "dispatch_async",
          [](artlite::Kernel& self, nb::handle ctrl_code, nb::list data_bufs) {
            auto refs = buffer_refs(data_bufs);
            self.dispatch_async(nb::cast<shim_xdna::bo&>(ctrl_code), refs);
            nb::handle py_self = nb::find(self);
            nb::object inflight = nb::getattr(py_self, "_inflight", nb::none());
            if (inflight.is_none()) {
              inflight = nb::list();
              nb::setattr(py_self, "_inflight", inflight);
            }
            nb::list l = nb::borrow<nb::list>(inflight);
            l.append(ctrl_code);
            l.append(data_bufs);
          },
          "ctrl_code"_a, "data_bufs"_a,
          "Submit a transaction and return (dispatch_wait waits for it).",
          nb::sig("def dispatch_async(self, ctrl_code: Buffer, data_bufs: list[Buffer]) -> None"))
      .def(
          "dispatch_wait",
          [](artlite::Kernel& self, uint32_t timeout_ms, int64_t spin_after_us, int spin_cpu, size_t keep) {
            retire(self, [&] { self.dispatch_wait(timeout_ms, spin_after_us, spin_cpu, keep); });
          },
          "timeout_ms"_a = 0, "spin_after_us"_a = -1, "spin_cpu"_a = 0, "keep"_a = 0,
          "Wait for every pending dispatch but the last ``keep`` (they run on, in order); raises DispatchError on a "
          "failure or when the timeout expires (0: none). spin_after_us >= 0: sleep until that long after the last "
          "asynchronous dispatch's submit, then poll on CPU spin_cpu (as dispatch does).",
          nb::sig("def dispatch_wait(self, timeout_ms: int = 0, spin_after_us: int = -1, spin_cpu: int = 0, "
                  "keep: int = 0) -> None"))
      .def("dispatch_poll", &artlite::Kernel::dispatch_poll,
           "Whether every pending dispatch has finished (reads the command states, no syscall).");

  // --- ParameterScratchpad ---
  // Direct binding to ParameterScratchpad (raw uint32_t* path).
  // The Python side passes a mapped BO viewed as uint32 and the params.txt path.
  nb::class_<ParameterScratchpad>(m, "ParameterScratchpad")
      .def(
          "__init__",
          [](ParameterScratchpad& self, nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>> buffer,
             const std::string& params_path) {
            new (&self) ParameterScratchpad(static_cast<uint32_t*>(buffer.data()), buffer.nbytes(),
                                            params_path);
          },
          "buffer"_a, "params_path"_a, nb::keep_alive<1, 2>())
      .def(
          "write",
          [](ParameterScratchpad& self, const std::string& name, uint32_t value) {
            self.write<uint32_t>(name, value);
          },
          "name"_a, "value"_a, nb::sig("def write(self, name: str, value: int) -> None"))
      .def(
          "sync",
          [](ParameterScratchpad& self, shim_xdna::bo& bo) {
            self.validate();
            bo.sync(shim_xdna::direction::host2device);
          },
          "bo"_a, nb::sig("def sync(self, bo: Buffer) -> None"));

  // --- Profiler ---
  nb::class_<artlite::Profiler>(m, "Profiler")
      .def(nb::init<shim_xdna::hw_ctx&, std::vector<std::pair<uint32_t, uint32_t>>,
                    std::vector<std::pair<uint32_t, std::string>>, std::string>(),
           "ctx"_a, "tiles"_a, "counters"_a, "out_path"_a, nb::keep_alive<1, 2>(),
           nb::sig("def __init__(self, ctx: HwContext, tiles: list[tuple[int, int]], "
                   "counters: list[tuple[int, str]], out_path: str) -> None"))
      .def("sample", &artlite::Profiler::sample)
      .def("record", &artlite::Profiler::record, "name"_a, "value"_a,
           "Append a host-side counter event (process 'host', one track per name).")
      .def("close", &artlite::Profiler::close);

  m.def("dump_buffer", &artlite::dump_bo, "buf"_a, "path"_a,
        "Sync the buffer device->host and write its raw contents to a file.");

  m.def(
      "gguf_strings_json",
      [](nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> data, size_t pos, uint64_t count,
         std::optional<nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>> ids) {
        std::string json;
        {
          nb::gil_scoped_release release;
          json = artlite::gguf_strings_json({data.data(), data.shape(0)}, pos, count,
                                            ids ? std::span<const int64_t>(ids->data(), ids->shape(0))
                                                : std::span<const int64_t>());
        }
        return nb::bytes(json.data(), json.size());
      },
      "data"_a, "pos"_a, "count"_a, "ids"_a = nb::none(),
      nb::sig("def gguf_strings_json(data: numpy.ndarray[numpy.uint8], pos: int, count: int, "
              "ids: numpy.ndarray[numpy.int64] | None = None) -> bytes"),
      "The JSON text of a GGUF metadata string array at byte pos of data (count x [u64 length, UTF-8 bytes]): "
      "without ids the array of the strings, with ids the object {string: id} of those strings.");

  m.def(
      "dequant_rows",
      [](std::string qtype, nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> raw,
         nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> out) {
        if (out.shape(0) != raw.shape(0)) throw artlite::Error("dequant_rows: out's rows are not raw's");
        nb::gil_scoped_release release;  // (a prompt's rows: the worker thread that prepares the next chunk's)
        artlite::dequant_rows(qtype, raw.data(), raw.shape(0), raw.shape(1), out.shape(1), out.data());
      },
      "qtype"_a, "raw"_a, "out"_a,
      nb::sig("def dequant_rows(qtype: str, raw: numpy.ndarray[numpy.uint8], out: numpy.ndarray[numpy.float32]) -> None"),
      "GGUF rows (raw: (n, row bytes) of a Q4_0, Q4_K, Q6_K or Q3_K tensor's blocks) dequantized into out (n, cols) fp32, "
      "its first cols columns: the values of gguf_model.QTensor.dequantize (its fp32 operations in their order).");
  m.def(
      "dequant_supported", [](std::string qtype) { return artlite::dequant_supported(qtype); }, "qtype"_a,
      "Whether dequant_rows reads the GGUF type.");

  m.def(
      "pack_tiles",
      [](nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> out, nb::list blocks, unsigned threads,
         bool release, std::optional<std::pair<std::string, nb::ndarray<const uint8_t, nb::ndim<1>, nb::device::cpu>>> file) {
        using Rows = nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
        using Raw = nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
        std::vector<artlite::PackBlock> bs;
        for (nb::handle h : blocks) {
          auto t = nb::cast<nb::tuple>(h);
          if (t.size() != 14) throw artlite::Error("pack_tiles: a block is a tuple of 14");
          artlite::PackBlock b{};
          b.fmt = artlite::tile_format(nb::cast<std::string>(t[0]));
          b.R = nb::cast<int>(t[1]);
          b.S = nb::cast<size_t>(t[2]);
          b.nsb = nb::cast<int>(t[3]);
          b.n_kc = nb::cast<int>(t[4]);
          b.kc_group = nb::cast<int>(t[5]);
          b.n_rt = nb::cast<int>(t[6]);
          b.rt0 = nb::cast<int>(t[7]);
          b.offset = nb::cast<size_t>(t[8]);
          b.col_stride = nb::cast<size_t>(t[9]);
          b.row_sb = nb::cast<int>(t[12]);
          b.G = nb::cast<int>(t[13]);
          const size_t row_bytes = size_t(b.row_sb) * artlite::block_bytes(b.fmt);
          for (nb::handle r : nb::cast<nb::list>(t[10])) {
            auto raw = nb::cast<Raw>(r, false);
            if (raw.shape(1) != row_bytes)
              throw artlite::Error(std::format("pack_tiles: tensor rows of {} bytes, the block's are {}",
                                               raw.shape(1), row_bytes));
            b.tensors.emplace_back(raw.data(), static_cast<int64_t>(raw.shape(0)));
          }
          auto rows = nb::cast<Rows>(t[11], false);
          b.rows = rows.data();
          b.cols = static_cast<int>(rows.shape(0));
          b.n = static_cast<int64_t>(rows.shape(1));
          bs.push_back(std::move(b));
        }
        artlite::PackOptions opt{threads, release};
        if (file) {
          opt.file = file->first;
          opt.map = file->second.data();
        }
        nb::gil_scoped_release release_gil;
        artlite::pack_tiles(out.data(), out.shape(0), bs, opt);
      },
      "out"_a, "blocks"_a, "threads"_a = 0, "release"_a = false, "file"_a = nb::none(),
      nb::sig("def pack_tiles(out: numpy.ndarray[numpy.uint8], blocks: list[tuple], threads: int = 0, "
              "release: bool = False, file: tuple[str, numpy.ndarray[numpy.uint8]] | None = None) -> None"),
      "Pack weight tiles (every tile format: bit-identical to the formats' pack_tile) into out. A block: "
      "(format, R, S, nsb, n_kc, kc_group, n_rt, rt0, offset, col_stride, tensors, rows, row_sb, G) as "
      "decode.image.TileImage.pack builds it (format: a tile format, BFP16 as \"BFP16/BF16\", \"BFP16/F16\" or "
      "\"BFP16/F32\" by its tensors' type; tensors: (rows, row bytes) uint8 arrays of GGUF blocks, stacked; rows: "
      "(columns, n) int64 stacked row indices, -1 a zero row (BFP16/F32: 2 i + p is plane p of row i); row_sb: a "
      "tensor row's super-blocks, at most n_kc nsb: zero super-blocks past them; G: tensor rows per tile row, "
      "consecutive in rows, along K). threads: 0 = one per physical core; release: "
      "MADV_DONTNEED each tensor's pages when done (file mappings only); file: (path, its whole mapping) of the "
      "tensors: read into the page cache ahead of the workers.");

  // --- The MoE prefill's host step (artlite/moe: plan, ctl) ---
  m.def(
      "moe_route",
      [](nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu> logits, int64_t n, int64_t row,
         int per_col, int64_t pitch, int E, nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> ids,
         nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> weights) {
        const int k = int(ids.shape(1));
        if (ids.shape(0) < size_t(n) || weights.shape(0) < size_t(n) || weights.shape(1) != ids.shape(1) ||
            (n && logits.shape(0) < size_t((n - 1) * row + (E / per_col - 1) * pitch + per_col)))
          throw artlite::Error("moe_route: arrays too small");
        nb::gil_scoped_release release;
        artlite::moe_route(logits.data(), n, row, per_col, pitch, E, k, ids.data(), weights.data());
      },
      "logits"_a, "n"_a, "row"_a, "per_col"_a, "pitch"_a, "E"_a, "ids"_a, "weights"_a,
      nb::sig("def moe_route(logits: numpy.ndarray[numpy.float32], n: int, row: int, per_col: int, pitch: int, E: int, "
              "ids: numpy.ndarray[numpy.int32], weights: numpy.ndarray[numpy.float32]) -> None"),
      "The top k of n tokens' E logits (token t's at logits[t row ..], expert e at (e / per_col) pitch + e % per_col) "
      "into ids and weights (n x k): artlite::moe_route.");
  m.def(
      "moe_plan",
      [](nb::ndarray<const int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> ids, int E, int chunk,
         nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> experts,
         nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> counts,
         nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> sizes,
         nb::ndarray<int32_t, nb::ndim<3>, nb::c_contig, nb::device::cpu> rows,
         nb::ndarray<int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> pos) {
        const int cols = int(experts.shape(0)), cap = int(experts.shape(1));
        if (counts.shape(0) != size_t(cols) || sizes.shape(0) != size_t(cols) || sizes.shape(1) != size_t(cap) ||
            rows.shape(0) != size_t(cols) || rows.shape(1) != size_t(cap) || rows.shape(2) != size_t(chunk) ||
            pos.shape(0) != ids.shape(0) || pos.shape(1) != ids.shape(1))
          throw artlite::Error("moe_plan: array shapes");
        nb::gil_scoped_release release;
        artlite::moe_plan(ids.data(), int64_t(ids.shape(0)), int(ids.shape(1)), E, cols, cap, chunk, experts.data(),
                          counts.data(), sizes.data(), rows.data(), pos.data());
      },
      "ids"_a, "E"_a, "chunk"_a, "experts"_a, "counts"_a, "sizes"_a, "rows"_a, "pos"_a,
      nb::sig("def moe_plan(ids: numpy.ndarray[numpy.int32], E: int, chunk: int, experts: numpy.ndarray[numpy.int32], "
              "counts: numpy.ndarray[numpy.int32], sizes: numpy.ndarray[numpy.int32], rows: numpy.ndarray[numpy.int32], "
              "pos: numpy.ndarray[numpy.int64]) -> None"),
      "The chunks of a routing (ids n x k) dealt to experts.shape[0] columns of experts.shape[1] chunks: "
      "artlite::moe_plan.");
  m.def(
      "ctl_assemble",
      [](nb::ndarray<uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> out, uint64_t base,
         nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> block0, nb::list templates,
         nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> blocks,
         nb::ndarray<const uint64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> bases,
         nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> tails,
         std::optional<nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>> lock_bits) {
        using Words = nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
        using Slots = nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
        std::vector<artlite::CtlTemplate> ts;
        for (nb::handle h : templates) {
          auto t = nb::cast<nb::tuple>(h);
          auto w = nb::cast<Words>(t[0], false);
          auto s = nb::cast<Slots>(t[1], false);
          if (s.shape(0) && s.shape(1) != 4) throw artlite::Error("ctl_assemble: slots of 4");
          ts.push_back({{w.data(), w.shape(0)}, {s.data(), s.shape(0) * (s.shape(0) ? 4 : 0)}});
        }
        const size_t n = blocks.shape(0);
        if (bases.shape(0) < n || (lock_bits && lock_bits->shape(0) < n))
          throw artlite::Error("ctl_assemble: a base row (and lock bits) per block");
        nb::gil_scoped_release release;
        return artlite::ctl_assemble(out.data(), out.shape(0), base, {block0.data(), block0.shape(0)}, ts,
                                     {blocks.data(), n}, bases.data(), int(bases.shape(1)),
                                     {tails.data(), tails.shape(0)}, lock_bits ? lock_bits->data() : nullptr);
      },
      "out"_a, "base"_a, "block0"_a, "templates"_a, "blocks"_a, "bases"_a, "tails"_a, "lock_bits"_a = nb::none(),
      nb::sig("def ctl_assemble(out: numpy.ndarray[numpy.uint32], base: int, block0: numpy.ndarray[numpy.uint32], "
              "templates: list[tuple[numpy.ndarray[numpy.uint32], numpy.ndarray[numpy.int64]]], "
              "blocks: numpy.ndarray[numpy.int32], bases: numpy.ndarray[numpy.uint64], tails: numpy.ndarray[numpy.uint32], "
              "lock_bits: numpy.ndarray[numpy.uint32] | None = None) -> int"),
      "A control stream into out (artlite::ctl_assemble): block 0, then block i from template blocks[i] (words, "
      "slots (S, 4): lo, hi (-1: lo alone), base index, offset) patched from bases[i], each with the tail that writes "
      "and pushes the next. Returns its words.");

  // --- Errors ---
  auto exc_error = nb::exception<artlite::Error>(m, "Error");
  nb::exception<artlite::ExecutableError>(m, "ExecutableError", exc_error);
  auto exc_dispatch = nb::exception<artlite::DispatchError>(m, "DispatchError", exc_error);
  // DispatchError carries .state (ert_cmd_state), .state_name and .health (the firmware's report
  // on a TIMEOUT as a dict, else None).
  nb::register_exception_translator(
      [](const std::exception_ptr& p, void* type) {
        try {
          std::rethrow_exception(p);
        } catch (const artlite::DispatchError& e) {
          nb::handle t(static_cast<PyObject*>(type));
          nb::object exc = t(e.what());
          exc.attr("state") = e.state;
          exc.attr("state_name") = std::string(shim_xdna::ert_cmd_state_name(e.state));
          if (e.health) {
            const auto& h = *e.health;
            nb::dict d;
            d["txn_op_idx"] = h.txn_op_idx;
            d["ctx_pc"] = h.ctx_pc;
            d["fatal_error_type"] = h.fatal_error_type;
            d["fatal_error_exception_type"] = h.fatal_error_exception_type;
            d["fatal_error_exception_pc"] = h.fatal_error_exception_pc;
            d["fatal_error_app_module"] = h.fatal_error_app_module;
            exc.attr("health") = d;
          } else {
            exc.attr("health") = nb::none();
          }
          PyErr_SetObject(t.ptr(), exc.ptr());
        }
      },
      exc_dispatch.ptr());
  m.def(
      "set_heap_reclaim",
      [](nb::callable f) {
        // (the function stays referenced for the process: a static's release would follow the interpreter's end)
        PyObject* fn = f.release().ptr();
        shim_xdna::set_heap_reclaim([fn] {
          nb::gil_scoped_acquire gil;
          return nb::cast<bool>(nb::handle(fn)());
        });
      },
      "f"_a,
      "f() is called when a DEV BO does not fit a device heap that cannot grow (Linux before 7.2): it frees DEV BOs "
      "the process can spare and returns True for the allocation to be tried once more.");
  // A full device heap (ENOSPC: a DEV BO that does not fit): HeapFullError, an Error.
  static PyObject* heap_full = PyErr_NewException("artlite._native.HeapFullError", exc_error.ptr(), nullptr);
  m.attr("HeapFullError") = nb::handle(heap_full);
  // A request the driver does not serve (EOPNOTSUPP: e.g. reading AIE tiles on Linux's own driver):
  // NotImplementedError, a RuntimeError.
  nb::register_exception_translator([](const std::exception_ptr& p, void*) {
    try {
      std::rethrow_exception(p);
    } catch (const std::system_error& e) {
      if (e.code().value() == ENOSPC) {
        PyErr_SetString(heap_full, e.what());
        return;
      }
      if (e.code().value() != EOPNOTSUPP) throw;
      PyErr_SetString(PyExc_NotImplementedError, e.what());
    }
  });
}
// NOLINTEND(bugprone-unused-raii)