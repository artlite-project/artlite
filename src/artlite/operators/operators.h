#pragma once

#include <aie_api/aie.hpp>

#define AIE_PRAGMA_STR(x) _Pragma(#x)
#define AIE_PRAGMA(x)     AIE_PRAGMA_STR(x)

// Active Loop Optimization Macros
#define AIE_LOOP_UNROLL(x)              AIE_PRAGMA(clang loop unroll_count(x))
#define AIE_LOOP_UNROLL_FULL            AIE_PRAGMA(clang loop unroll(full))
#define AIE_LOOP_NO_UNROLL              AIE_PRAGMA(clang loop unroll(disable))
#define AIE_LOOP_MIN_ITERATION_COUNT(x) AIE_PRAGMA(clang loop min_iteration_count(x))
#define AIE_LOOP_MAX_ITERATION_COUNT(x) AIE_PRAGMA(clang loop max_iteration_count(x))
#define AIE_TRY_INITIATION_INTERVAL(x)  AIE_PRAGMA(clang loop pipeline_initiation_interval(x))
#define AIE_PREPARE_FOR_POSTPIPELINING  AIE_PRAGMA(clang loop pipeline(disable))
#define AIE_LOOP_HINT(k, v)             AIE_PRAGMA(clang loop hint(k, v))

// WAW register rewriter hints (AIEWawRegRewriter)
#define AIE_LOOP_GPR_REALLOC            AIE_LOOP_HINT(aie-gpr-realloc, 1)
#define AIE_LOOP_REALLOC_LOOPAWARE      AIE_LOOP_HINT(aie-realloc-loopaware, 1)
#define AIE_LOOP_AGGRESSIVE_REALLOC     AIE_LOOP_HINT(aie-aggressive-realloc, 1)
#define AIE_LOOP_WAW_MIN_LAT(x)         AIE_LOOP_HINT(aie-waw-reg-rewrite-min-lat, x)
#define AIE_LOOP_REALLOC_II_BIAS(x)     AIE_LOOP_HINT(aie-realloc-ii-bias, x)
// Enum ordinals of aie-reg-rewrite-mode (metadata only accepts the integer)
#define AIE_LOOP_REG_REWRITE_BASIC        AIE_LOOP_HINT(aie-reg-rewrite-mode, 0)
#define AIE_LOOP_REG_REWRITE_LATENCYAWARE AIE_LOOP_HINT(aie-reg-rewrite-mode, 2)
#define AIE_LOOP_REG_REWRITE_SWPAWARE     AIE_LOOP_HINT(aie-reg-rewrite-mode, 3)
#define AIE_LOOP_REG_REWRITE_SWPAWARE_AUTO AIE_LOOP_HINT(aie-reg-rewrite-mode, 4)

#include "kernel_lib.h"  // al:: helpers shared by the operators' templates (after the macros: they use them)

// Epilogue WAR rewriter (AIEEpilogueRegRewriter), mainly for
// outer-loop-pipelined epilogue blocks
#define AIE_LOOP_OLP_WAR_RENAME         AIE_LOOP_HINT(aie-olp-war-rename, 1)

// Loop-aware inter-block scheduling: ignore memory deps for this loop
#define AIE_LOOP_IGNORE_MEM_DEPS        AIE_LOOP_HINT(aie-safe-to-ignore-memory-deps, 1)

// Inner loop versioning (AIEInnerLoopVersioning): clone into a pipelined
// high-trip-count copy and a verbatim low-trip-count copy behind a runtime guard
#define AIE_LOOP_VERSIONING             AIE_LOOP_HINT(aie-loop-versioning, 1)

// Outer loop pipeliner (AIEOuterLoopPipeliner); place on the outer loop
#define AIE_LOOP_OUTER_PIPELINING        AIE_LOOP_HINT(aie-enable-outer-loop-pipelining, 1)
#define AIE_LOOP_NO_OUTER_PIPELINING     AIE_LOOP_HINT(aie-enable-outer-loop-pipelining, 0)
#define AIE_LOOP_OUTER_MIN_TRIP_COUNT(x) AIE_LOOP_HINT(aie-outer-loop-pipelining-min-trip-count, x)
#define AIE_LOOP_OUTER_NO_SPLIT_PROLOGUE AIE_LOOP_HINT(aie-outer-loop-pipelining-split-prologue, 0)
#define AIE_LOOP_OUTER_LEAN_STAGE0       AIE_LOOP_HINT(aie-outer-loop-pipelining-lean-stage0, 1)
#define AIE_LOOP_OUTER_SPECULATIVE       AIE_LOOP_HINT(aie-outer-loop-pipelining-speculative, 1)
#define AIE_LOOP_OUTER_HW_LOOP           AIE_LOOP_HINT(aie-outer-loop-hw-loop, 1)

#define AIE_LOOP_RANGE(a, ...) \
    AIE_LOOP_MIN_ITERATION_COUNT(a) __VA_OPT__(AIE_LOOP_MAX_ITERATION_COUNT(__VA_ARGS__))
