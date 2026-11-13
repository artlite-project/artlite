#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cerrno>
#include <exception>
#include <string>
#include <system_error>

#include "artlite/error.h"
#include "artlite/shim.h"
#include "bindings/bindings.h"

using namespace nb::literals;

namespace artlite::bindings {

void bind_errors(nb::module_& m) {
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

}  // namespace artlite::bindings
