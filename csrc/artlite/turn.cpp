#include "artlite/turn.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <random>
#include <string>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#endif

namespace artlite::turn {

namespace {

constexpr int64_t kTurn = 0, kWaiting = 1, kOwner = 8, kSlot0 = 64, kSlots = 256;  // (the lock file's bytes: turn.h)

enum class Mode { kShared, kExclusive, kUnlock };

#ifdef _WIN32
using Handle = HANDLE;
const Handle kNone = INVALID_HANDLE_VALUE;

Handle open_file() {
  wchar_t dir[MAX_PATH + 1];
  const DWORD n = ::GetTempPathW(MAX_PATH + 1, dir);
  if (n == 0 || n > MAX_PATH) return kNone;
  const std::wstring path = std::wstring(dir, n) + L"artlite-npu.lock";
  return ::CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                       nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
}

void close_file(Handle h) { ::CloseHandle(h); }

long process_id() { return static_cast<long>(::GetCurrentProcessId()); }

// Bytes [at, at + n). false: another process's lock is in the way (wait = false) or the call failed (conflict():
// which)
bool lock(Handle h, int64_t at, Mode mode, bool wait, int64_t n = 1) {
  OVERLAPPED ov{};
  ov.Offset = static_cast<DWORD>(at);
  const DWORD len = static_cast<DWORD>(n);
  if (mode == Mode::kUnlock) return ::UnlockFileEx(h, 0, len, 0, &ov) != 0;
  const DWORD flags = (mode == Mode::kExclusive ? LOCKFILE_EXCLUSIVE_LOCK : 0) | (wait ? 0 : LOCKFILE_FAIL_IMMEDIATELY);
  return ::LockFileEx(h, flags, 0, len, 0, &ov) != 0;
}

bool conflict() { return ::GetLastError() == ERROR_LOCK_VIOLATION; }

uint64_t read_owner(Handle h) {
  OVERLAPPED ov{};
  ov.Offset = kOwner;
  uint64_t v = 0;
  DWORD n = 0;
  return ::ReadFile(h, &v, sizeof v, &n, &ov) && n == sizeof v ? v : 0;
}

void write_owner(Handle h, uint64_t v) {
  OVERLAPPED ov{};
  ov.Offset = kOwner;
  DWORD n = 0;
  ::WriteFile(h, &v, sizeof v, &n, &ov);
}
#else
using Handle = int;
constexpr Handle kNone = -1;

Handle open_file() {
  const char* tmp = std::getenv("TMPDIR");
  for (const std::string dir : {"/dev/shm", tmp && *tmp ? tmp : "/tmp"}) {
    const int fd = ::open((dir + "/artlite-npu.lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    if (fd >= 0) {
      ::fchmod(fd, 0666);  // (past the umask: other users' processes open it too; fails on another user's file)
      return fd;
    }
  }
  return kNone;
}

void close_file(Handle fd) { ::close(fd); }

long process_id() { return static_cast<long>(::getpid()); }

// Open-file-description locks of bytes [at, at + n): a forked child's copy of the descriptor shares them
// (State::own reopens).
bool lock(Handle fd, int64_t at, Mode mode, bool wait, int64_t n = 1) {
  struct flock l{};
  l.l_type = mode == Mode::kShared ? F_RDLCK : mode == Mode::kExclusive ? F_WRLCK : F_UNLCK;
  l.l_whence = SEEK_SET;
  l.l_start = at;
  l.l_len = n;
  while (::fcntl(fd, wait ? F_OFD_SETLKW : F_OFD_SETLK, &l) != 0)
    if (errno != EINTR) return false;
  return true;
}

bool conflict() { return errno == EAGAIN || errno == EACCES; }

uint64_t read_owner(Handle fd) {
  uint64_t v = 0;
  return ::pread(fd, &v, sizeof v, kOwner) == sizeof v ? v : 0;
}

void write_owner(Handle fd, uint64_t v) {
  if (::pwrite(fd, &v, sizeof v, kOwner) != sizeof v) return;  // (a lost id: the next holder counts one turn more)
}
#endif

struct State {
  std::mutex m;
  size_t held = 0;
  std::atomic<uint64_t> foreign{0};
  std::atomic<int64_t> waited_ns{0};
  uint64_t id = 0;
  long pid = -1;
  Handle h = kNone;
  int64_t slot = kSlots;  // (its byte kSlot0 + slot; kSlots: none free)

  // The lock file of this process: opened at the first call, and again in a forked child (which would share the
  // parent's locks), with a holder id and a slot of its own.
  void own() {
    const long p = process_id();
    if (p == pid) return;
    if (h != kNone) close_file(h);
    h = open_file();
    pid = p;
    held = 0;
    foreign = 0;
    std::random_device rd;
    id = (static_cast<uint64_t>(rd()) << 32 | rd()) | 1;  // (0: no holder yet)
    slot = 0;
    while (h != kNone && slot < kSlots && !lock(h, kSlot0 + slot, Mode::kExclusive, false)) slot++;
  }

  // Whether another process holds a byte of [at, at + n).
  bool taken(int64_t at, int64_t n = 1) {
    if (n <= 0) return false;
    if (lock(h, at, Mode::kExclusive, false, n)) {
      lock(h, at, Mode::kUnlock, false, n);
      return false;
    }
    return conflict();
  }

  // Whether another process waits for the turn (holds byte kWaiting).
  bool waiting() { return taken(kWaiting); }
};

State& state() {
  static State s;
  return s;
}

}  // namespace

void enter() {
  State& s = state();
  std::lock_guard g(s.m);
  s.own();
  if (s.held++ > 0 || s.h == kNone) return;
  const auto t0 = std::chrono::steady_clock::now();
  if (!lock(s.h, kTurn, Mode::kExclusive, false)) {  // (another process's turn: wait for it, as a waiter)
    lock(s.h, kWaiting, Mode::kShared, true);
    lock(s.h, kTurn, Mode::kExclusive, true);
    lock(s.h, kWaiting, Mode::kUnlock, false);
    s.waited_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
                              .count(),
                          std::memory_order_relaxed);
  }
  if (read_owner(s.h) != s.id) {
    s.foreign.fetch_add(1, std::memory_order_relaxed);
    write_owner(s.h, s.id);
  }
}

void leave(size_t n) {
  State& s = state();
  std::lock_guard g(s.m);
  if (n == 0 || s.held == 0) return;
  s.held -= std::min(n, s.held);
  if (s.held == 0 && s.h != kNone) lock(s.h, kTurn, Mode::kUnlock, false);
}

uint64_t foreign() { return state().foreign.load(std::memory_order_relaxed); }

double waited() { return 1e-9 * static_cast<double>(state().waited_ns.load(std::memory_order_relaxed)); }

bool contended() {
  State& s = state();
  std::lock_guard g(s.m);
  s.own();
  return s.h != kNone && s.waiting();
}

bool peers() {
  State& s = state();
  std::lock_guard g(s.m);
  s.own();
  return s.h != kNone && (s.taken(kSlot0, s.slot) || s.taken(kSlot0 + s.slot + 1, kSlots - s.slot - 1));
}

void pass() {
  State& s = state();
  {
    std::lock_guard g(s.m);
    s.own();
    if (s.held > 0 || s.h == kNone) return;
  }
  const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(kPassMs);
  while (std::chrono::steady_clock::now() < end) {
    {
      std::lock_guard g(s.m);
      if (s.held > 0 || !s.waiting() || read_owner(s.h) != s.id) return;  // (a waiter has the turn now)
    }
    std::this_thread::sleep_for(std::chrono::microseconds(20));
  }
}

}  // namespace artlite::turn
