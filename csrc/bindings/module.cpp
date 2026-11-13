#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdlib>
#include <string>

#include "bindings/bindings.h"

extern const char artlite_source_hash[];  // (source_hash.cpp, generated: CMakeLists.txt's ARTLITE_SOURCE_HASH)

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

  // Zero-copy numpy views of Buffer mappings (map()) keep their
  // Buffer/Device wrappers alive until after nanobind's exit-time leak audit,
  // producing false "leaked N instances" reports even though gc finds zero
  // live instances at teardown and the actual munmap/GEM-close runs in ~bo().
  // No memory is lost when the process exits. Suppress the spurious audit;
  // ARTLITE_NB_LEAK_WARNINGS=1 re-enables it when hunting real leaks.
  if (!std::getenv("ARTLITE_NB_LEAK_WARNINGS"))
    nb::set_leak_warnings(false);

  namespace b = artlite::bindings;
  b::bind_device(m);
  b::bind_dispatch(m);
  b::bind_scratchpad(m);
  b::bind_profiler(m);
  b::bind_gguf(m);
  b::bind_sampling(m);
  b::bind_pack(m);
  b::bind_moe(m);
  b::bind_errors(m);
}
// NOLINTEND(bugprone-unused-raii)
