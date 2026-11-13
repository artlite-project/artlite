#pragma once

// The Python module's bindings, a function per domain (NB_MODULE in module.cpp calls them in this order).

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace artlite::bindings {

/// The NPU's turn between processes, Buffer, HwContext, Executable, Device (the shim's objects).
void bind_device(nb::module_& m);
/// Kernel: dispatches, asynchronous ones and their waits.
void bind_dispatch(nb::module_& m);
/// ParameterScratchpad.
void bind_scratchpad(nb::module_& m);
/// Profiler and dump_buffer.
void bind_profiler(nb::module_& m);
/// GGUF strings, row dequantization, Q8_0 quantization.
void bind_gguf(nb::module_& m);
/// Sampling: top_logits, pick, max_prob, logsumexp.
void bind_sampling(nb::module_& m);
/// The weight packers: pack_tiles, pack_slots.
void bind_pack(nb::module_& m);
/// The MoE prefill's host step: moe_route, moe_plan, ctl_assemble.
void bind_moe(nb::module_& m);
/// The exception types and their translators, the heap reclaim hook (after every function that raises them).
void bind_errors(nb::module_& m);

}  // namespace artlite::bindings
