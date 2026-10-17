#include "artlite/kernel.h"

#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <format>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "artlite/error.h"
#include "shim_xdna/bo.h"
#include "shim_xdna/device.h"
#include "shim_xdna/ert.h"
#include "shim_xdna/hwctx.h"

namespace artlite {

using namespace shim_xdna;

namespace {

constexpr size_t kCmdBoSize = 4096;
constexpr int kMaxSpinMs = 20;  // (a dispatch longer than expected: then the blocking wait)
constexpr int kNapUs = 200;     // (the sleep before the poll in naps: a long one lets the CPU's core drop into a deep
                                // C-state, and the host work after the dispatch then runs on a cold core, ~10x slower)

uint32_t state_of(const bo& cmd) {
  return std::atomic_ref<uint32_t>(*static_cast<uint32_t*>(cmd.map())).load(std::memory_order_acquire) &
         ERT_STATE_MASK;
}

}  // namespace

struct Kernel::Impl {
  struct Command {
    std::unique_ptr<bo> cmd;
    uint64_t seq;
    uint32_t opcode;
  };

  device& dev;
  hw_ctx& ctx;
  uint32_t cu_mask;
  std::vector<std::unique_ptr<bo>> idle;  // exec buffers to reuse
  std::deque<Command> pending;            // submitted, in timeline order
  std::vector<uint32_t> arg_handles;
  std::chrono::steady_clock::time_point t_submit;  // the last dispatch_async's

  Impl(device& d, hw_ctx& c, std::string_view name)
      : dev(d), ctx(c), cu_mask(1u << c.cu_index(name.empty() ? c.m_cu_info[0].m_name : name)) {}

  // Waits for what is still running: the driver does not keep a job's buffers alive.
  ~Impl() {
    if (pending.empty()) return;
    try {
      ctx.wait(pending.back().seq, 0);
    } catch (...) {
    }
  }

  // Fill an exec buffer (header, CU mask, payload) and submit it. Every word the driver reads is
  // rewritten, which also restores a buffer whose payload the driver wiped after a failure.
  void submit(uint32_t opcode, std::span<const uint32_t> payload, const bo& first,
              std::span<const std::reference_wrapper<bo>> bufs) {
    if (payload.size() + 2 > kCmdBoSize / sizeof(uint32_t))
      throw DispatchError(std::format("command payload of {} words does not fit the exec buffer", payload.size()));
    std::unique_ptr<bo> cmd;
    if (idle.empty()) {
      cmd = std::make_unique<bo>(dev.get_pdev(), kCmdBoSize, static_cast<uint32_t>(XCL_BO_FLAGS_EXECBUF));
    } else {
      cmd = std::move(idle.back());
      idle.pop_back();
    }
    auto* w = static_cast<uint32_t*>(cmd->map());
    w[1] = cu_mask;
    std::copy(payload.begin(), payload.end(), w + 2);
    // NEW before the submit: the driver resets the state itself only when the job starts
    w[0] = ert_header(ERT_CMD_STATE_NEW, static_cast<uint32_t>(payload.size() + 1), opcode, ERT_CU);

    arg_handles.clear();
    arg_handles.push_back(first.get_drm_bo_handle());
    for (auto& b : bufs) arg_handles.push_back(b.get().get_drm_bo_handle());
    uint64_t seq;
    try {
      seq = ctx.submit(*cmd, arg_handles);
    } catch (...) {
      idle.push_back(std::move(cmd));
      throw;
    }
    pending.push_back({std::move(cmd), seq, opcode});
  }

  DispatchError failure(const Command& c, uint32_t state) const {
    const char* what = c.opcode == ERT_START_NPU_PREEMPT_ELF ? "ELF" : "transaction";
    std::string msg = std::format("NPU {} execution failed: {} (state {})", what, ert_cmd_state_name(state), state);
    std::optional<CommandHealth> health;
    if (state == ERT_CMD_STATE_TIMEOUT) {
      ert_ctx_health_aie2 r;
      std::memcpy(&r, static_cast<const uint32_t*>(c.cmd->map()) + 1, sizeof(r));
      if (r.version == 1 && r.npu_gen == 0) {
        health = CommandHealth{r.txn_op_idx, r.ctx_pc, r.fatal_error_type, r.fatal_error_exception_type,
                               r.fatal_error_exception_pc, r.fatal_error_app_module};
        msg += std::format(
            "; firmware report: txn_op_idx {}, ctx_pc {:#x}, fatal error type {:#x} (exception type {:#x}, "
            "pc {:#x}, app module {:#x})",
            r.txn_op_idx, r.ctx_pc, r.fatal_error_type, r.fatal_error_exception_type, r.fatal_error_exception_pc,
            r.fatal_error_app_module);
      } else {
        msg += "; no firmware report";
      }
      msg += " (the driver's hang detection stopped it: dmesg has 'TDR timeout detected')";
    } else if (state == ERT_CMD_STATE_ABORT) {
      msg += " (cancelled by the recovery from a stall)";
    }
    return DispatchError(msg, state, health);
  }

  // Retire the finished commands, all of them after a completed wait; throws the first failure.
  void reap() {
    std::optional<DispatchError> first;
    while (!pending.empty()) {
      Command& c = pending.front();
      uint32_t state = state_of(*c.cmd);
      if (!ert_cmd_state_is_terminal(state)) {
        ctx.wait(c.seq, 0);  // (in order: done when a later one is)
        state = state_of(*c.cmd);
      }
      if (state != ERT_CMD_STATE_COMPLETED && !first) {
        first = ert_cmd_state_is_terminal(state)
                    ? failure(c, state)
                    : DispatchError(std::format("NPU command finished without a final state ({}): the driver could "
                                                "not run it",
                                                ert_cmd_state_name(state)),
                                    state, std::nullopt);
      }
      idle.push_back(std::move(c.cmd));
      pending.pop_front();
    }
    if (first) throw *first;
  }

  void wait_all(uint32_t timeout_ms) {
    if (pending.empty()) return;
    const Command& last = pending.back();
    uint32_t state = state_of(*last.cmd);
    if (!ert_cmd_state_is_terminal(state) && !ctx.wait(last.seq, timeout_ms)) {
      state = state_of(*last.cmd);
      throw DispatchError(std::format("NPU command did not finish within {} ms (state {}); it keeps running "
                                      "and cannot be cancelled",
                                      timeout_ms, ert_cmd_state_name(state)),
                          state, std::nullopt);
    }
    reap();
  }

  bool done() const {
    return std::all_of(pending.begin(), pending.end(),
                       [](const Command& c) { return ert_cmd_state_is_terminal(state_of(*c.cmd)); });
  }

  // Sleep until t0 + after_us on CPU cpu (in naps of kNapUs, ending early when the commands finish: an after_us
  // from a slow run must not hold the next ones that long, each run's own time feeding the next estimate), then poll
  // the commands' states there (yielding: the completion's work may need that CPU) until they finish or kMaxSpinMs
  // pass.
  void spin(std::chrono::steady_clock::time_point t0, int64_t after_us, int cpu) const {
    cpu_set_t old, one;
    if (cpu < 0 || cpu >= CPU_SETSIZE || sched_getaffinity(0, sizeof old, &old) != 0 || !CPU_ISSET(cpu, &old)) return;
    CPU_ZERO(&one);
    CPU_SET(cpu, &one);
    if (sched_setaffinity(0, sizeof one, &one) != 0) return;
    const auto until = t0 + std::chrono::microseconds(after_us);
    const auto nap = std::chrono::microseconds(kNapUs);
    while (std::chrono::steady_clock::now() + nap < until && !done()) std::this_thread::sleep_for(nap);
    if (!done()) std::this_thread::sleep_until(until);
    const auto limit = std::chrono::steady_clock::now() + std::chrono::milliseconds(kMaxSpinMs);
    while (!done() && std::chrono::steady_clock::now() < limit) sched_yield();
    sched_setaffinity(0, sizeof old, &old);
  }

  // ERT_START_CU payload (transactions):
  //   regmap[0-1]: firmware opcode (uint64, value 3)
  //   regmap[2-3]: instruction buffer BO (device address, 64-bit)
  //   regmap[4]:   instruction count (uint32, number of uint32s)
  //   regmap[5..]: data buffer BOs (device address, 64-bit each)
  void submit_cu(bo& ctrl_code, std::span<const std::reference_wrapper<bo>> data_bufs) {
    static constexpr uint32_t kFirmwareOpcode = 3;
    uint32_t payload[kCmdBoSize / sizeof(uint32_t)];
    size_t n = 0;
    auto put64 = [&](uint64_t v) {
      payload[n++] = static_cast<uint32_t>(v);
      payload[n++] = static_cast<uint32_t>(v >> 32);
    };
    if (5 + 2 * data_bufs.size() > std::size(payload) - 2)
      throw DispatchError(std::format("{} data buffers do not fit the exec buffer", data_bufs.size()));
    put64(kFirmwareOpcode);
    put64(ctrl_code.get_paddr());
    payload[n++] = static_cast<uint32_t>(ctrl_code.size() / sizeof(uint32_t));
    for (auto& b : data_bufs) put64(b.get().get_paddr());
    submit(ERT_START_CU, {payload, n}, ctrl_code, data_bufs);
  }

  // ERT_START_NPU_PREEMPT_ELF payload (amdxdna_cmd_preempt_data): instruction buffer, save and
  // restore buffers (none), their sizes in bytes, property count.
  // instr_bytes: the control code's length (0: the whole BO; a reused BO holds pieces of several lengths).
  void submit_elf(bo& instr_bo, std::span<const std::reference_wrapper<bo>> data_bufs, size_t instr_bytes = 0) {
    uint64_t inst = instr_bo.get_paddr();
    if (instr_bytes > instr_bo.size())
      throw DispatchError(std::format("{} bytes of control code in a BO of {}", instr_bytes, instr_bo.size()));
    const uint32_t payload[10] = {static_cast<uint32_t>(inst), static_cast<uint32_t>(inst >> 32), 0, 0, 0, 0,
                                  static_cast<uint32_t>(instr_bytes ? instr_bytes : instr_bo.size()), 0, 0, 0};
    submit(ERT_START_NPU_PREEMPT_ELF, payload, instr_bo, data_bufs);
  }
};

Kernel::Kernel(device& dev, hw_ctx& ctx) : impl_(std::make_unique<Impl>(dev, ctx, std::string_view{})) {}

Kernel::Kernel(device& dev, hw_ctx& ctx, std::string_view kernel_name)
    : impl_(std::make_unique<Impl>(dev, ctx, kernel_name)) {}

Kernel::~Kernel() = default;
Kernel::Kernel(Kernel&&) noexcept = default;
Kernel& Kernel::operator=(Kernel&&) noexcept = default;

void Kernel::dispatch(bo& ctrl_code, std::initializer_list<std::reference_wrapper<bo>> data_bufs,
                      uint32_t timeout_ms) {
  dispatch(ctrl_code, std::span<const std::reference_wrapper<bo>>(data_bufs.begin(), data_bufs.size()), timeout_ms);
}

void Kernel::dispatch(bo& ctrl_code, std::span<const std::reference_wrapper<bo>> data_bufs, uint32_t timeout_ms) {
  impl_->submit_cu(ctrl_code, data_bufs);
  impl_->wait_all(timeout_ms);
}

void Kernel::dispatch(bo& ctrl_code, std::span<const std::reference_wrapper<bo>> data_bufs, uint32_t timeout_ms,
                      int64_t spin_after_us, int spin_cpu) {
  const auto t0 = std::chrono::steady_clock::now();
  impl_->submit_cu(ctrl_code, data_bufs);
  if (spin_after_us >= 0) impl_->spin(t0, spin_after_us, spin_cpu);
  impl_->wait_all(timeout_ms);
}

void Kernel::dispatch_elf(bo& instr_bo, std::span<const std::reference_wrapper<bo>> data_bufs, uint32_t timeout_ms,
                          size_t instr_bytes) {
  impl_->submit_elf(instr_bo, data_bufs, instr_bytes);
  impl_->wait_all(timeout_ms);
}

void Kernel::dispatch_elf_async(bo& instr_bo, std::span<const std::reference_wrapper<bo>> data_bufs,
                                size_t instr_bytes) {
  impl_->submit_elf(instr_bo, data_bufs, instr_bytes);
}

void Kernel::dispatch_async(bo& ctrl_code, std::span<const std::reference_wrapper<bo>> data_bufs) {
  impl_->t_submit = std::chrono::steady_clock::now();
  impl_->submit_cu(ctrl_code, data_bufs);
}

void Kernel::dispatch_wait(uint32_t timeout_ms) { impl_->wait_all(timeout_ms); }

void Kernel::dispatch_wait(uint32_t timeout_ms, int64_t spin_after_us, int spin_cpu) {
  if (spin_after_us >= 0) impl_->spin(impl_->t_submit, spin_after_us, spin_cpu);
  impl_->wait_all(timeout_ms);
}

bool Kernel::dispatch_poll() const { return impl_->done(); }

}  // namespace artlite
