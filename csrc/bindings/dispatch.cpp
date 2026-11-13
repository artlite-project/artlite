#include <nanobind/nanobind.h>
#include <nanobind/stl/string_view.h>

#include <cstdint>
#include <functional>
#include <vector>

#include "artlite/kernel.h"
#include "artlite/shim.h"
#include "bindings/bindings.h"

using namespace nb::literals;

namespace {

std::vector<std::reference_wrapper<shim_xdna::bo>> buffer_refs(nb::list data_bufs) {
  std::vector<std::reference_wrapper<shim_xdna::bo>> refs;
  refs.reserve(nb::len(data_bufs));
  for (nb::handle b : data_bufs) refs.emplace_back(nb::cast<shim_xdna::bo&>(b));
  return refs;
}

// The buffer references of the asynchronous dispatches: a pending one keeps its buffers referenced (in _inflight)
// until a wait retires it, the driver does not keep them alive.
void keep_inflight(artlite::Kernel& self, nb::handle first, nb::list data_bufs) {
  nb::handle py_self = nb::find(self);
  nb::object inflight = nb::getattr(py_self, "_inflight", nb::none());
  if (inflight.is_none()) {
    inflight = nb::list();
    nb::setattr(py_self, "_inflight", inflight);
  }
  nb::list l = nb::borrow<nb::list>(inflight);
  l.append(first);
  l.append(data_bufs);
}

// Run a blocking dispatch or wait without the GIL; afterwards drop the buffer references of the
// async dispatches it retired (all of them unless a wait timed out).
template <typename F>
void retire(artlite::Kernel& self, F&& f) {
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

}  // namespace

namespace artlite::bindings {

void bind_dispatch(nb::module_& m) {
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
            auto& bo = nb::cast<shim_xdna::bo&>(instr_bo);
            {
              nb::gil_scoped_release release;  // (the NPU's turn may be another process's)
              self.dispatch_elf_async(bo, refs, instr_bytes);
            }
            keep_inflight(self, instr_bo, data_bufs);
          },
          "instr_bo"_a, "data_bufs"_a, "instr_bytes"_a = 0,
          nb::sig("def dispatch_elf_async(self, instr_bo: Buffer, data_bufs: list[Buffer], "
                  "instr_bytes: int = 0) -> None"))
      .def(
          "dispatch_async",
          [](artlite::Kernel& self, nb::handle ctrl_code, nb::list data_bufs) {
            auto refs = buffer_refs(data_bufs);
            auto& bo = nb::cast<shim_xdna::bo&>(ctrl_code);
            {
              nb::gil_scoped_release release;  // (the NPU's turn may be another process's)
              self.dispatch_async(bo, refs);
            }
            keep_inflight(self, ctrl_code, data_bufs);
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
}

}  // namespace artlite::bindings
