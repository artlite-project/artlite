// Kernel on Windows: transactions (and full ELFs' control programs, which are transactions) run as XRT runs of the
// program's xclbin kernel.
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <deque>
#include <format>
#include <string>
#include <thread>
#include <vector>

#include "artlite/error.h"
#include "artlite/kernel.h"
#include "artlite/shim.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

namespace artlite {

using shim_xdna::bo;
using shim_xdna::device;
using shim_xdna::hw_ctx;

namespace {

constexpr int kMaxSpinMs = 20;  // (a dispatch longer than expected: then the blocking wait)
constexpr int kNapUs = 200;
// The kernel's arguments (aiecc's xclbin): opcode, instruction buffer, its length in words, then the buffers.
constexpr unsigned kArgOpcode = 0, kArgInstr = 1, kArgWords = 2, kArgFirstBuffer = 3;
constexpr uint64_t kFirmwareOpcode = 3;
constexpr size_t kMaxBuffers = 5;  // (bo0 .. bo4: a kernel with more fails to run)

uint32_t state_of(const xrt::run& r) { return static_cast<uint32_t>(r.state()); }

}  // namespace

struct Kernel::Impl {
  hw_ctx& ctx;
  std::unique_ptr<xrt::kernel> named;  // (a kernel other than the context's)
  xrt::kernel* kernel;
  // (runs and their buffer count: a run is reused only for as many buffers, so that it holds no earlier one)
  std::vector<std::pair<xrt::run, size_t>> idle;
  std::deque<std::pair<xrt::run, size_t>> pending;  // submitted, in order
  std::chrono::steady_clock::time_point t_submit;

  Impl(hw_ctx& c, std::string_view name) : ctx(c), kernel(c.m_kernel.get()) {
    if (!name.empty() && name != c.m_kernel_name) {
      named = std::make_unique<xrt::kernel>(*c.m_ctx, std::string(name));
      kernel = named.get();
    }
  }

  ~Impl() {
    for (auto& [r, n] : pending) {
      try {
        r.wait();
      } catch (...) {
      }
    }
  }

  // The NPU reaches only the XRT buffers its command lists, at most kMaxBuffers, even where the transaction holds the
  // addresses. Up to kMaxBuffers data buffers are the arguments themselves, in order (an insts.bin's DDR patches take
  // the k-th one's address); more, or `resolved` (the transaction has every address and no DDR patch), are their
  // regions, each once.
  // instr_bytes: the transaction's length (0: the whole BO).
  void submit_cu(bo& ctrl_code, std::span<const std::reference_wrapper<bo>> data_bufs, size_t instr_bytes = 0,
                 bool resolved = false) {
    if (instr_bytes > ctrl_code.size())
      throw DispatchError(std::format("{} bytes of control code in a BO of {}", instr_bytes, ctrl_code.size()));
    try {
      std::vector<xrt::bo*> args;
      for (size_t i = 0; i < data_bufs.size(); i++) {
        bo& b = data_bufs[i].get();
        const unsigned arg = kArgFirstBuffer + static_cast<unsigned>(std::min(i, kMaxBuffers - 1));
        if (!resolved && data_bufs.size() <= kMaxBuffers) {
          args.push_back(&b.xrt_for(ctx, arg));
        } else if (b.is_data()) {
          xrt::bo* x = &b.region_for(ctx, arg);
          if (std::find(args.begin(), args.end(), x) == args.end()) args.push_back(x);
        }
      }
      if (args.size() > kMaxBuffers)
        throw DispatchError(std::format("the transaction's buffers lie in {} XRT buffers: a command takes {}",
                                        args.size(), kMaxBuffers));
      xrt::run run;
      const auto same = std::find_if(idle.begin(), idle.end(), [&](const auto& p) { return p.second == args.size(); });
      if (same == idle.end()) {
        run = xrt::run(*kernel);
      } else {
        run = std::move(same->first);
        idle.erase(same);
      }
      run.set_arg(kArgOpcode, kFirmwareOpcode);
      run.set_arg(kArgInstr, ctrl_code.xrt_for(ctx, kArgInstr));
      run.set_arg(kArgWords, static_cast<uint32_t>((instr_bytes ? instr_bytes : ctrl_code.size()) / sizeof(uint32_t)));
      for (size_t i = 0; i < args.size(); i++) run.set_arg(kArgFirstBuffer + static_cast<unsigned>(i), *args[i]);
      run.start();
      pending.emplace_back(std::move(run), args.size());
    } catch (const DispatchError&) {
      throw;
    } catch (const std::exception& e) {
      throw DispatchError(std::format("submitting an NPU transaction failed: {}", e.what()));
    }
  }

  // A full ELF's control program: a transaction with every address resolved (BDs, PDI loads, scratchpad). Its
  // instruction buffers (PDIs, scratchpad) are the context's.
  void submit_elf(bo& instr, std::span<const std::reference_wrapper<bo>> bufs, size_t instr_bytes) {
    submit_cu(instr, bufs, instr_bytes, true);
  }

  // XRT's account of a failed run (on a timeout: the firmware's report), as its exception says it.
  static std::string account(xrt::run& r) {
    try {
      r.wait2();
    } catch (const std::exception& e) {
      return e.what();
    }
    return {};
  }

  DispatchError failure(xrt::run& r, uint32_t state) const {
    std::string msg = std::format("NPU transaction execution failed: {} (state {})",
                                  shim_xdna::ert_cmd_state_name(state), state);
    if (const std::string why = account(r); !why.empty()) msg += "; " + why;
    return DispatchError(msg, state, std::nullopt);
  }

  // Retire the finished runs, all of them after a completed wait; throws the first failure.
  void reap() {
    std::optional<DispatchError> first;
    while (!pending.empty()) {
      xrt::run& r = pending.front().first;
      uint32_t state = state_of(r);
      if (state != shim_xdna::ERT_CMD_STATE_COMPLETED && !first) first = failure(r, state);
      idle.push_back(std::move(pending.front()));
      pending.pop_front();
    }
    if (first) throw *first;
  }

  // One deadline for all the runs (timeout_ms 0: none). XRT's wait returns TIMEOUT for an expired wait and may return
  // late (the Windows timer's 15.6 ms), so each run's own state decides.
  void wait_all(uint32_t timeout_ms) {
    if (pending.empty()) return;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (auto& [r, n] : pending) {
      uint32_t state = state_of(r);
      const auto left =
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
      try {
        if (!shim_xdna::ert_cmd_state_is_terminal(state) && (timeout_ms == 0 || left.count() > 0))
          r.wait(timeout_ms == 0 ? std::chrono::milliseconds(0) : left);  // (XRT: 0 ms blocks)
      } catch (const std::exception& e) {
        state = state_of(r);
        if (shim_xdna::ert_cmd_state_is_terminal(state)) continue;  // (reap reports it)
        throw DispatchError(std::format("waiting for an NPU transaction failed: {}", e.what()), state, std::nullopt);
      }
      state = state_of(r);
      if (!shim_xdna::ert_cmd_state_is_terminal(state))
        throw DispatchError(std::format("NPU command did not finish within {} ms (state {}); it keeps running and "
                                        "cannot be cancelled",
                                        timeout_ms, shim_xdna::ert_cmd_state_name(state)),
                            state, std::nullopt);
    }
    reap();
  }

  bool done() const {
    return std::all_of(pending.begin(), pending.end(),
                       [](const auto& p) { return shim_xdna::ert_cmd_state_is_terminal(state_of(p.first)); });
  }

  // Sleep until t0 + after_us on CPU cpu (in naps, ending early when the runs finish), then poll their states there
  // until they finish or kMaxSpinMs pass: that CPU is awake when the completion comes.
  void spin(std::chrono::steady_clock::time_point t0, int64_t after_us, int cpu) const {
    if (cpu < 0 || cpu >= 64) return;
    const DWORD_PTR old = ::SetThreadAffinityMask(::GetCurrentThread(), DWORD_PTR(1) << cpu);
    if (!old) return;
    const auto until = t0 + std::chrono::microseconds(after_us);
    const auto nap = std::chrono::microseconds(kNapUs);
    while (std::chrono::steady_clock::now() + nap < until && !done()) std::this_thread::sleep_for(nap);
    if (!done()) std::this_thread::sleep_until(until);
    const auto limit = std::chrono::steady_clock::now() + std::chrono::milliseconds(kMaxSpinMs);
    while (!done() && std::chrono::steady_clock::now() < limit) ::SwitchToThread();
    ::SetThreadAffinityMask(::GetCurrentThread(), old);
  }
};

Kernel::Kernel(device&, hw_ctx& ctx) : impl_(std::make_unique<Impl>(ctx, std::string_view{})) {}

Kernel::Kernel(device&, hw_ctx& ctx, std::string_view kernel_name) : impl_(std::make_unique<Impl>(ctx, kernel_name)) {}

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
