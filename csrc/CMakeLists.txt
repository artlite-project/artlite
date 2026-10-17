add_library(artlite_shim STATIC
  shim_xdna/device.cpp
  shim_xdna/bo.cpp
  shim_xdna/hwctx.cpp
)
target_include_directories(artlite_shim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shim_xdna)
target_link_libraries(artlite_shim PUBLIC PkgConfig::LIBDRM)

add_library(artlite STATIC
  artlite/dequant.cpp
  artlite/executable.cpp
  artlite/file_io.cpp
  artlite/gguf_json.cpp
  artlite/kernel.cpp
  artlite/profiler.cpp
  artlite/tile_pack.cpp
)
# the weight packers' AVX-512 VBMI / GFNI code (Artlite runs on Zen 5 only)
set_source_files_properties(artlite/tile_pack.cpp PROPERTIES COMPILE_OPTIONS -march=znver5)
# (the embedding rows' fp32 operations as gguf_model's QTensor.dequantize orders them: no fused multiply-adds)
set_source_files_properties(artlite/dequant.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
target_include_directories(artlite PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(artlite PUBLIC artlite_shim)
