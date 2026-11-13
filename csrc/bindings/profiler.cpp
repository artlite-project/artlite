#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "artlite/profiler.h"
#include "artlite/shim.h"
#include "bindings/bindings.h"

using namespace nb::literals;

namespace artlite::bindings {

void bind_profiler(nb::module_& m) {
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
}

}  // namespace artlite::bindings
