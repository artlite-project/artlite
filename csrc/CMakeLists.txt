# What every backend shares (dequantization, executables, GGUF strings, weight packing, the MoE prefill's host step)
add_library(artlite_core OBJECT
  artlite/dequant.cpp
  artlite/executable.cpp
  artlite/gguf_json.cpp
  artlite/moe_host.cpp
  artlite/tile_pack.cpp
)
target_include_directories(artlite_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(MSVC)
  set_source_files_properties(artlite/tile_pack.cpp artlite/moe_host.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX512)
  # (the embedding rows' fp32 operations as gguf_model's QTensor.dequantize orders them: no fused multiply-adds)
  set_source_files_properties(artlite/dequant.cpp PROPERTIES COMPILE_OPTIONS /fp:precise)
else()
  # the weight packers' AVX-512 VBMI / GFNI code, the MoE routing's (Artlite runs on Zen 5 only)
  set_source_files_properties(artlite/tile_pack.cpp artlite/moe_host.cpp PROPERTIES COMPILE_OPTIONS -march=znver5)
  set_source_files_properties(artlite/dequant.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
endif()

# A backend: its shim, the Kernel on it and the code over its buffers and contexts (file reads, the profiler)
set(ARTLITE_ON_SHIM artlite/file_io.cpp artlite/profiler.cpp $<TARGET_OBJECTS:artlite_core>)

if(NOT WIN32)  # the amdxdna driver's ioctls
  add_library(artlite_xdna STATIC
    shim_xdna/device.cpp
    shim_xdna/bo.cpp
    shim_xdna/hwctx.cpp
    artlite/kernel.cpp
    ${ARTLITE_ON_SHIM}
  )
  target_include_directories(artlite_xdna PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shim_xdna)
  target_link_libraries(artlite_xdna PUBLIC PkgConfig::LIBDRM)
endif()

if(XRT_DIR)  # XRT's API (the Ryzen AI software's xrt_coreutil on Windows)
  add_library(artlite_xrt STATIC
    shim_xrt/device.cpp
    shim_xrt/bo.cpp
    shim_xrt/hwctx.cpp
    shim_xrt/xclbin.cpp
    artlite/kernel_xrt.cpp
    ${ARTLITE_ON_SHIM}
  )
  target_include_directories(artlite_xrt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${XRT_INCLUDE_DIR})
  target_compile_definitions(artlite_xrt PUBLIC ARTLITE_SHIM_XRT ARTLITE_XRT_DIR="${XRT_DIR}")
  if(WIN32)
    target_compile_definitions(artlite_xrt PUBLIC DISABLE_ABI_CHECK=1)
  endif()
  target_link_libraries(artlite_xrt PUBLIC ${XRT_COREUTIL_LIBRARY})
endif()
