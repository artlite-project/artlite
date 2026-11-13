#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/unique_ptr.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "artlite/error.h"
#include "artlite/executable.h"
#include "artlite/file_io.h"
#include "artlite/shim.h"
#include "artlite/turn.h"
#include "bindings/bindings.h"

using namespace nb::literals;

namespace {

std::atomic<size_t> user_bo_bytes{size_t(2) << 20};  // (alloc_host_buffer: from this size on, alloc_user_bo)

}  // namespace

namespace artlite::bindings {

void bind_device(nb::module_& m) {
  m.def(
      "set_user_bo_bytes", [](size_t nbytes) { user_bo_bytes = nbytes; }, "nbytes"_a,
      "Device.alloc_host_buffer's buffers from this size on are user BOs (2 MB pages; 2 MB by default), smaller ones "
      "the driver's (4 KB pages): artlite-probe --diagnose measures both.");
  m.def("user_bo_bytes", [] { return size_t(user_bo_bytes); }, "set_user_bo_bytes' size.");

  // --- The NPU's turn between processes (turn.h) ---
  m.def(
      "turn_enter",
      [] {
        nb::gil_scoped_release release;
        artlite::turn::enter();
      },
      "Until this process holds the NPU's turn (nested holds count, each pending dispatch's among them): no other "
      "process's dispatch runs meanwhile. turn_leave gives a hold back.");
  m.def("turn_leave", [] { artlite::turn::leave(); }, "Give back a turn_enter hold.");
  m.def("turn_foreign", &artlite::turn::foreign,
        "The turns this process took after another process's: a change means another process's hardware context "
        "may have run on the NPU since.");
  m.def("turn_waited", &artlite::turn::waited,
        "Seconds this process waited for another process's NPU turn to end (in its lifetime): a timing of its own "
        "work subtracts it.");
  m.def("turn_contended", &artlite::turn::contended, "Whether another process waits for the NPU's turn.");
  m.def("turn_peers", &artlite::turn::peers,
        "Whether another process uses the NPU (any of these functions called, the process not ended): its turns may "
        "come between this process's dispatches.");
  m.def(
      "turn_pass",
      [] {
        nb::gil_scoped_release release;
        artlite::turn::pass();
      },
      "Holding nothing: until a process waiting for the NPU's turn has taken it (at most 50 ms), so that this "
      "process's next dispatch comes after its turn.");

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
      .def(
          "sync_lines",
          [](shim_xdna::bo& self, nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig> offsets) {
            self.sync_lines(offsets.data(), offsets.size());
          },
          "offsets"_a, "Host writes of 64-byte cache lines (their byte offsets) -> DDR.")
      .def_prop_ro("paddr", [](shim_xdna::bo& self) -> uint64_t { return self.get_paddr(); })
      .def(
          "paddr_in", [](shim_xdna::bo& self, const shim_xdna::hw_ctx& ctx) -> uint64_t { return self.paddr_in(ctx); },
          "ctx"_a,
          "The device address in a hardware context (on XRT an instruction buffer is made in each context's memory group: "
          "its own address in each).");

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
            if (size >= user_bo_bytes) {
              nb::gil_scoped_release release;
              return self.alloc_user_bo(size);
            }
            return self.alloc_bo(size, static_cast<uint32_t>(XCL_BO_FLAGS_HOST_ONLY));
          },
          "size"_a, nb::keep_alive<0, 1>(),
          "A zeroed host buffer the NPU reads and writes; from 2 MB on (set_user_bo_bytes) in this process's "
          "memory, as alloc_user_buffer (the driver's own BOs have 4 KB pages, and DMA rows on pages of their own run at a "
          "fraction of the rate: on Strix Halo 12 instead of 57 GB/s for 128-byte rows).")
      .def(
          "alloc_user_buffer",
          [](shim_xdna::device& self, size_t size) {
            nb::gil_scoped_release release;
            return self.alloc_user_bo(size);
          },
          "size"_a, nb::keep_alive<0, 1>(),
          "A zeroed host buffer in this process's memory (2 MB pages faulted in by several threads, pinned by the "
          "driver): for large buffers (16 GB: ~0.3 s, a driver's BO 1.7 s). Like alloc_host_buffer otherwise.")
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
      .def(
          "get_info",
          [](const shim_xdna::device& self, uint32_t param, nb::bytes query) {
            std::vector<uint8_t> buf(static_cast<const uint8_t*>(query.data()),
                                     static_cast<const uint8_t*>(query.data()) + query.size());
            size_t n = self.get_info(param, buf);
            return nb::bytes(reinterpret_cast<const char*>(buf.data()), std::min(n, buf.size()));
          },
          "param"_a, "query"_a,
          "The driver's GET_INFO answer for param (DRM_AMDXDNA_QUERY_*) in a buffer of query's size that starts as "
          "query (the input of queries that take one: telemetry's type), cut to the size the driver reports "
          "(artlite-probe --diagnose: sensors, firmware telemetry). NotImplementedError on XRT.")
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
}

}  // namespace artlite::bindings
