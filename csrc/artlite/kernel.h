#pragma once

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <span>
#include <string_view>

namespace shim_xdna {
struct bo;
struct device;
struct hw_ctx;
}  // namespace shim_xdna

namespace artlite {

/// Submits commands to a hardware context: on the amdxdna shim (kernel.cpp) the firmware's command packets in
/// pooled exec-buffer BOs, so a dispatch costs one submit ioctl plus the wait; on XRT (kernel_xrt.cpp) runs of the
/// context's kernel. Any number of asynchronous dispatches may be pending (the driver runs at most 4 per context; a
/// 5th submit blocks until one completes). Failures throw DispatchError with the command state (and the firmware's
/// report on a TIMEOUT). The caller keeps the buffers of a pending dispatch alive until it is waited for (the driver
/// does not); the destructor waits for pending dispatches.
class Kernel {
 public:
  Kernel(shim_xdna::device& dev, shim_xdna::hw_ctx& ctx);
  Kernel(shim_xdna::device& dev, shim_xdna::hw_ctx& ctx, std::string_view kernel_name);
  ~Kernel();

  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;
  Kernel(Kernel&&) noexcept;
  Kernel& operator=(Kernel&&) noexcept;

  // Dispatch: ctrl_code buffer contains NPU instructions,
  // data_buffers are input/output buffers bound as kernel arguments.
  void dispatch(shim_xdna::bo& ctrl_code,
                std::initializer_list<std::reference_wrapper<shim_xdna::bo>> data_bufs,
                uint32_t timeout_ms = 0);

  void dispatch(shim_xdna::bo& ctrl_code,
                std::span<const std::reference_wrapper<shim_xdna::bo>> data_bufs,
                uint32_t timeout_ms = 0);

  // dispatch, the calling thread sleeping spin_after_us after the submit (or until the command finishes, if earlier),
  // then polling the command's state on CPU spin_cpu (the one that takes the NPU's completion interrupts) until it
  // finishes (at most kMaxSpinMs, then the blocking wait): that CPU is awake when the interrupt comes (spin.h).
  // spin_after_us < 0, or spin_cpu outside the thread's affinity: a plain dispatch.
  void dispatch(shim_xdna::bo& ctrl_code,
                std::span<const std::reference_wrapper<shim_xdna::bo>> data_bufs,
                uint32_t timeout_ms, int64_t spin_after_us, int spin_cpu);

  // Dispatch a full ELF's control program: one command with ERT_START_NPU_PREEMPT_ELF (on XRT: a transaction, its
  // DDR patches removed: txn.without_ddr_patches).
  // instr_bo: patched instruction buffer (ctrltext with addresses resolved)
  // data_bufs: the BOs the ELF uses (passed to drivers that pin them)
  // instr_bytes: the control code's length from the BO's start (0: the whole BO)
  void dispatch_elf(shim_xdna::bo& instr_bo,
                    std::span<const std::reference_wrapper<shim_xdna::bo>> data_bufs,
                    uint32_t timeout_ms = 0, size_t instr_bytes = 0);

  // Asynchronous dispatch_elf: submit the command and return immediately (its submit time is the one dispatch_wait's
  // spin counts from).
  void dispatch_elf_async(shim_xdna::bo& instr_bo,
                          std::span<const std::reference_wrapper<shim_xdna::bo>> data_bufs,
                          size_t instr_bytes = 0);

  // Asynchronous dispatch of a transaction: submit the command and return immediately (its submit time is the one
  // dispatch_wait's spin counts from).
  void dispatch_async(shim_xdna::bo& ctrl_code, std::span<const std::reference_wrapper<shim_xdna::bo>> data_bufs);

  // Wait for every pending dispatch; throws DispatchError if one failed (the first failure) or
  // the timeout expired (timeout_ms 0: none; the commands then stay pending).
  void dispatch_wait(uint32_t timeout_ms = 0);

  // dispatch_wait, first sleeping until spin_after_us after the last asynchronous dispatch's submit and polling on CPU
  // spin_cpu (as the spinning dispatch does); keep: return once at most that many dispatches are pending (the last
  // ones submitted: they run on after the wait, in order).
  void dispatch_wait(uint32_t timeout_ms, int64_t spin_after_us, int spin_cpu, size_t keep = 0);

  // Whether every pending dispatch has finished (no syscall: reads the command states).
  bool dispatch_poll() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace artlite
