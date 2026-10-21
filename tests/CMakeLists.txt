# Python (the venv's, with Peano and aiecc) is found by the top-level CMakeLists.txt.
set(PEANO_DIR "${Python_SITELIB}/llvm-aie")
find_program(AIECC aiecc HINTS "${Python_ROOT_DIR}/bin" REQUIRED)
# Same flags as ArtcOperator.compile() (artlite/operators/base.py: build, CORE_LLC_FLAGS).
set(AIECC_FLAGS
  --unified
  --get-pdi --get-npu-insts
  --peano=${PEANO_DIR}
  -Xllc=--enable-aie-zol-without-minitercount=false
  --pdi-name=main.pdi
  --npu-insts-name=insts.bin
)
set(TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")

# add_aie_target(<name> <mlir_src> [EXTRA_DEPS dep1 ...])
# Compiles MLIR -> PDI + insts.
# Output: ${TEST_DATA_DIR}/<name>/main.pdi and ${TEST_DATA_DIR}/<name>/insts.bin
function(add_aie_target name mlir_src)
  cmake_parse_arguments(ARG "" "" "EXTRA_DEPS" ${ARGN})
  set(out_dir "${TEST_DATA_DIR}/${name}")
  set(pdi_file "${out_dir}/main.pdi")
  set(insts_file "${out_dir}/insts.bin")
  file(MAKE_DIRECTORY ${out_dir})

  add_custom_command(
    OUTPUT ${pdi_file} ${insts_file}
    COMMAND ${AIECC} ${AIECC_FLAGS} ${mlir_src}
    WORKING_DIRECTORY ${out_dir}
    DEPENDS ${mlir_src} ${ARG_EXTRA_DEPS}
    COMMENT "MLIR -> PDI+insts: ${name}"
  )
  add_custom_target(${name}_aie DEPENDS ${pdi_file} ${insts_file})
endfunction()

# add_mlir_gen(<name> <script>)
# Generates MLIR by running a Python script; output: ${TEST_DATA_DIR}/<name>.mlir
function(add_mlir_gen name script)
  set(mlir_file "${TEST_DATA_DIR}/${name}.mlir")
  add_custom_command(
    OUTPUT ${mlir_file}
    COMMAND Python::Interpreter ${script} > ${mlir_file}
    DEPENDS ${script}
    COMMENT "MLIR gen: ${name}"
  )
  add_custom_target(${name}_mlir DEPENDS ${mlir_file})
endfunction()

# --- Test data ---

add_mlir_gen(addone "${DATA_DIR}/addone.py")
add_aie_target(addone "${TEST_DATA_DIR}/addone.mlir" EXTRA_DEPS addone_mlir)

# --- C++ test binaries ---

add_executable(artlite_tests npu_test.cpp)
target_link_libraries(artlite_tests PRIVATE artlite_xdna GTest::gtest GTest::gtest_main)
target_compile_definitions(artlite_tests PRIVATE
  TEST_DATA_DIR="${TEST_DATA_DIR}"
)
add_dependencies(artlite_tests addone_aie)
add_test(NAME npu_tests COMMAND artlite_tests)
