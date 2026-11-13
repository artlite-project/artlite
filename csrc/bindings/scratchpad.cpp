#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <new>
#include <string>

#include "artlite/scratchpad.h"
#include "artlite/shim.h"
#include "bindings/bindings.h"

using namespace nb::literals;

namespace artlite::bindings {

void bind_scratchpad(nb::module_& m) {
  // --- ParameterScratchpad ---
  // Direct binding to ParameterScratchpad (raw uint32_t* path).
  // The Python side passes a mapped BO viewed as uint32 and the params.txt path.
  nb::class_<artlite::ParameterScratchpad>(m, "ParameterScratchpad")
      .def(
          "__init__",
          [](artlite::ParameterScratchpad& self, nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>> buffer,
             const std::string& params_path) {
            new (&self) artlite::ParameterScratchpad(static_cast<uint32_t*>(buffer.data()), buffer.nbytes(),
                                            params_path);
          },
          "buffer"_a, "params_path"_a, nb::keep_alive<1, 2>())
      .def(
          "write",
          [](artlite::ParameterScratchpad& self, const std::string& name, uint32_t value) {
            self.write<uint32_t>(name, value);
          },
          "name"_a, "value"_a, nb::sig("def write(self, name: str, value: int) -> None"))
      .def(
          "sync",
          [](artlite::ParameterScratchpad& self, shim_xdna::bo& bo) {
            self.validate();
            bo.sync(shim_xdna::direction::host2device);
          },
          "bo"_a, nb::sig("def sync(self, bo: Buffer) -> None"));
}

}  // namespace artlite::bindings
