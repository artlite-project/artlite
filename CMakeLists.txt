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
if(WIN32)
  # XRT's headers and xrt_coreutil.lib from $XILINX_XRT, set for the build only (set at run time, XRT looks for its
  # driver library there); its directory is recorded in the module for the tools (xclbinutil, aiebu-asm)
  find_path(XRT_INCLUDE_DIR xrt/xrt_device.h HINTS "$ENV{XILINX_XRT}/include")
  find_library(XRT_COREUTIL_LIBRARY xrt_coreutil HINTS "$ENV{XILINX_XRT}/lib")
  if(NOT XRT_INCLUDE_DIR OR NOT XRT_COREUTIL_LIBRARY)
    message(FATAL_ERROR "XRT not found: set XILINX_XRT to its install directory, which holds include/ and lib/ "
                        "(usually C:\\Xilinx\\XRT)")
  endif()
  get_filename_component(XRT_DIR "${XRT_INCLUDE_DIR}" DIRECTORY)
  if(MSVC)
    add_compile_options(/Zc:__cplusplus /utf-8)  # (XRT's headers pick std::any by __cplusplus)
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBDRM REQUIRED IMPORTED_TARGET libdrm)
endif()

# --- C++ libraries (artlite_shim + artlite) ---
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

nanobind_add_module(_native STABLE_ABI csrc/bindings.cpp)
target_link_libraries(_native PRIVATE artlite)
if(WIN32)
  target_compile_definitions(_native PRIVATE ARTLITE_XRT_DIR="${XRT_DIR}")
endif()

install(TARGETS _native LIBRARY DESTINATION artlite)

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
