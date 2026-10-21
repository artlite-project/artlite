#pragma once

// The end of a dispatch on the CPU that takes the NPU's completion interrupts (both Kernels: kernel.cpp,
// kernel_xrt.cpp). A CPU in its deepest C-state, 350 us exit latency on Strix, delays each completion by 0.3 - 0.8 ms:
// a dispatch of ~1 ms after the host idled took 1.9 instead of 1.2.
#include <chrono>
#include <cstdint>
#include <thread>

#include "artlite/platform.h"

namespace artlite {

inline constexpr int kMaxSpinMs = 20;  // (a dispatch longer than expected: then the blocking wait)
inline constexpr int kNapUs = 200;     // (the sleep before the poll in naps: a long one lets the CPU's core drop into a
                                       // deep C-state, and the host work after the dispatch then runs on a cold core,
                                       // ~10x slower)

// Sleep until t0 + after_us on CPU cpu (in naps of kNapUs, ending early when done(): an after_us from a slow run must
// not hold the next ones that long, each run's own time feeding the next estimate), then poll done() there (yielding:
// the completion's work may need that CPU) until it holds or kMaxSpinMs pass. A CPU outside the thread's affinity:
// returns at once.
template <typename Done>
void spin_on_cpu(std::chrono::steady_clock::time_point t0, int64_t after_us, int cpu, Done done) {
  platform::ThreadOnCpu on(cpu);
  if (!on.pinned()) return;
  const auto until = t0 + std::chrono::microseconds(after_us);
  const auto nap = std::chrono::microseconds(kNapUs);
  while (std::chrono::steady_clock::now() + nap < until && !done()) platform::nap(nap);
  if (!done())
    platform::nap(std::chrono::duration_cast<std::chrono::microseconds>(until - std::chrono::steady_clock::now()));
  const auto limit = std::chrono::steady_clock::now() + std::chrono::milliseconds(kMaxSpinMs);
  while (!done() && std::chrono::steady_clock::now() < limit) std::this_thread::yield();
}

}  // namespace artlite
