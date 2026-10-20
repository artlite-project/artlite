if(WIN32)
  # XRT (the Ryzen AI software's xrt_coreutil) replaces the amdxdna driver's ioctls
  add_library(artlite_shim STATIC
    shim_xrt/device.cpp
    shim_xrt/bo.cpp
    shim_xrt/hwctx.cpp
  )
  target_include_directories(artlite_shim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${XRT_INCLUDE_DIR})
  target_compile_definitions(artlite_shim PUBLIC "NOMINMAX=" WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS
                                                 DISABLE_ABI_CHECK=1)
  target_link_libraries(artlite_shim PUBLIC ${XRT_COREUTIL_LIBRARY})
  set(ARTLITE_KERNEL_SRC artlite/kernel_xrt.cpp)
else()
  add_library(artlite_shim STATIC
    shim_xdna/device.cpp
    shim_xdna/bo.cpp
    shim_xdna/hwctx.cpp
  )
  target_include_directories(artlite_shim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shim_xdna)
  target_link_libraries(artlite_shim PUBLIC PkgConfig::LIBDRM)
  set(ARTLITE_KERNEL_SRC artlite/kernel.cpp)
endif()

add_library(artlite STATIC
  artlite/dequant.cpp
  artlite/executable.cpp
  artlite/file_io.cpp
  artlite/gguf_json.cpp
  ${ARTLITE_KERNEL_SRC}
  artlite/profiler.cpp
  artlite/tile_pack.cpp
)
# the weight packers' AVX-512 VBMI / GFNI code (Artlite runs on Zen 5 only)
if(MSVC)
  set_source_files_properties(artlite/tile_pack.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX512)
  # (the embedding rows' fp32 operations as gguf_model's QTensor.dequantize orders them: no fused multiply-adds)
  set_source_files_properties(artlite/dequant.cpp PROPERTIES COMPILE_OPTIONS /fp:precise)
else()
  set_source_files_properties(artlite/tile_pack.cpp PROPERTIES COMPILE_OPTIONS -march=znver5)
  set_source_files_properties(artlite/dequant.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
endif()
target_include_directories(artlite PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(artlite PUBLIC artlite_shim)
