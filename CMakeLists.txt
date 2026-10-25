cmake_minimum_required(VERSION 3.15...3.27)
project(artlite LANGUAGES CXX)

option(ARTLITE_BUILD_TESTS "Build C++ tests" OFF)

if(NOT SKBUILD)
  if(ARTLITE_BUILD_TESTS)
    message(STATUS "Standalone configure for the C++ tests")
  else()
    message(WARNING "\
      This CMake file is meant to be executed using 'scikit-build-core': \
      uv pip install --no-build-isolation -ve .  \
      Standalone configures are for the C++ tests only (separate tree, venv active): \
      cmake -S . -B build-tests -DARTLITE_BUILD_TESTS=ON")
  endif()
  # Standalone: use the project venv's Python (and its nanobind) when present.
  if(EXISTS "${CMAKE_SOURCE_DIR}/.venv")
    set(Python_ROOT_DIR "${CMAKE_SOURCE_DIR}/.venv")
  endif()
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- System dependencies ---
# The backends (src/artlite/backend.py): Artlite's shim on the amdxdna driver's ioctls (Linux, libdrm) and the same
# runtime over XRT's API (xrt_coreutil): Windows' only one; on Linux a second module where XRT is found.
option(ARTLITE_XRT "Linux: also build the XRT backend (artlite._native_xrt) where XRT is installed" ON)
if(NOT WIN32)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBDRM REQUIRED IMPORTED_TARGET libdrm)
endif()
if(WIN32 OR ARTLITE_XRT)
  # XRT's headers and xrt_coreutil from $XILINX_XRT (on Windows set for the build only: set at run time, XRT looks for
  # its driver library there), else the system's; its directory is recorded in the module for its tools (aiebu-asm)
  find_path(XRT_INCLUDE_DIR xrt/xrt_device.h HINTS "$ENV{XILINX_XRT}/include" /opt/xilinx/xrt/include)
  find_library(XRT_COREUTIL_LIBRARY xrt_coreutil HINTS "$ENV{XILINX_XRT}/lib" /opt/xilinx/xrt/lib)
  if(XRT_INCLUDE_DIR AND XRT_COREUTIL_LIBRARY)
    get_filename_component(XRT_DIR "${XRT_INCLUDE_DIR}" DIRECTORY)
  elseif(WIN32)
    message(FATAL_ERROR "XRT not found: set XILINX_XRT to its install directory, which holds include/ and lib/ "
                        "(usually C:\\Xilinx\\XRT)")
  else()
    message(STATUS "XRT not found: no XRT backend (artlite._native_xrt)")
  endif()
endif()
if(MSVC)
  add_compile_options(/Zc:__cplusplus /utf-8)  # (XRT's headers pick std::any by __cplusplus)
endif()
if(WIN32)  # every source that includes windows.h (platform.h): no min / max macros
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()

# --- C++ libraries (a static library per backend: artlite_xdna, artlite_xrt) ---
add_subdirectory(csrc)

# --- Python + nanobind ---
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module
             OPTIONAL_COMPONENTS Development.SABIModule)
# scikit-build-core puts nanobind on the prefix path; standalone configures
# ask the Python environment where its CMake package lives.
if(NOT SKBUILD AND NOT nanobind_ROOT)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
endif()
find_package(nanobind CONFIG REQUIRED)

# A module per backend: the platform's own as _native, Linux's XRT one as _native_xrt (its own nanobind domain: the
# same C++ types)
function(artlite_module name backend)
  nanobind_add_module(${name} STABLE_ABI ${ARGN} csrc/bindings.cpp)
  target_link_libraries(${name} PRIVATE artlite_${backend})
  target_compile_definitions(${name} PRIVATE ARTLITE_MODULE=${name})
  install(TARGETS ${name} LIBRARY DESTINATION artlite)
endfunction()
if(WIN32)
  artlite_module(_native xrt)
else()
  artlite_module(_native xdna)
  if(XRT_DIR)
    artlite_module(_native_xrt xrt NB_DOMAIN xrt)
  endif()
endif()

# --- Stub generation ---
nanobind_add_stub(
    _native_stub
    MODULE _native
    OUTPUT "${CMAKE_SOURCE_DIR}/src/artlite/_native.pyi"
    PYTHON_PATH $<TARGET_FILE_DIR:_native>
    DEPENDS _native
    MARKER_FILE "${CMAKE_SOURCE_DIR}/src/artlite/py.typed"
)

# --- C++ tests (opt-in) ---
if(ARTLITE_BUILD_TESTS)
  find_package(GTest REQUIRED)
  enable_testing()
  add_subdirectory(tests)
endif()
