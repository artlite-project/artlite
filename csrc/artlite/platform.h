#pragma once

// The few operating-system calls of the host code (file reads, aligned memory, page hints, sleeps, CPU affinity), for
// Linux and Windows.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <system_error>
#include <thread>

#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#endif

// A function that may use AVX-512 although the file is not compiled for it (MSVC needs no attribute).
#if defined(_MSC_VER) && !defined(__clang__)
#define ARTLITE_AVX512_FN
#else
#define ARTLITE_AVX512_FN __attribute__((target("avx512f")))
#endif

namespace artlite::platform {

inline size_t page_size() {
#ifdef _WIN32
  SYSTEM_INFO si;
  ::GetSystemInfo(&si);
  return si.dwPageSize;
#else
  return static_cast<size_t>(::sysconf(_SC_PAGESIZE));
#endif
}

inline void* aligned_alloc(size_t align, size_t n) {
#ifdef _WIN32
  return ::_aligned_malloc(n, align);
#else
  return std::aligned_alloc(align, n);
#endif
}

struct AlignedFree {
  void operator()(void* p) const {
#ifdef _WIN32
    ::_aligned_free(p);
#else
    std::free(p);
#endif
  }
};

template <typename T>
using AlignedPtr = std::unique_ptr<T, AlignedFree>;

template <typename T = char>
AlignedPtr<T> aligned_buffer(size_t align, size_t n) {
  return AlignedPtr<T>(static_cast<T*>(aligned_alloc(align, n)));
}

// Tells the system a mapping's whole pages inside [addr, addr + n) are no longer needed (MADV_DONTNEED; Windows keeps them).
inline void drop_pages(void* addr, size_t n) {
#ifndef _WIN32
  ::madvise(addr, n, MADV_DONTNEED);
#else
  (void)addr;
  (void)n;
#endif
}

// Whether a read should bring [addr, addr + n) of a file mapping into memory: some page is not there (Linux: mincore;
// a failed query says no). Windows has no such query and always says yes.
inline bool needs_reading([[maybe_unused]] void* addr, [[maybe_unused]] size_t n, [[maybe_unused]] unsigned char* vec) {
#ifdef _WIN32
  return true;
#else
  return ::mincore(addr, n, vec) == 0 &&
         !std::all_of(vec, vec + n / page_size(), [](unsigned char r) { return r & 1; });
#endif
}

// Sleep about `d`. On Windows on a high-resolution waitable timer: Sleep() and std::this_thread::sleep_for wait for
// the system timer's ticks, 15.6 ms unless a process asked for finer ones.
inline void nap(std::chrono::microseconds d) {
  if (d.count() <= 0) return;
#ifdef _WIN32
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
  struct Timer {
    HANDLE h = ::CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    ~Timer() {
      if (h) ::CloseHandle(h);
    }
  };
  thread_local Timer timer;
  LARGE_INTEGER due;
  due.QuadPart = -static_cast<LONGLONG>(d.count()) * 10;  // (100 ns units; negative: from now)
  if (timer.h && ::SetWaitableTimerEx(timer.h, &due, 0, nullptr, nullptr, nullptr, 0)) {
    ::WaitForSingleObject(timer.h, INFINITE);
    return;
  }
#endif
  std::this_thread::sleep_for(d);
}

// The calling thread runs on CPU `cpu` while this lives, if its affinity allows that CPU (`pinned()`; else nothing
// changes); the affinity is restored at the end.
class ThreadOnCpu {
 public:
  explicit ThreadOnCpu(int cpu) {
#ifdef _WIN32
    if (cpu < 0 || cpu >= 64) return;
    const DWORD_PTR one = DWORD_PTR(1) << cpu;
    old_ = ::SetThreadAffinityMask(::GetCurrentThread(), one);
    if (old_ && !(old_ & one)) {
      ::SetThreadAffinityMask(::GetCurrentThread(), old_);
      old_ = 0;
    }
#else
    cpu_set_t one;
    if (cpu < 0 || cpu >= CPU_SETSIZE || ::sched_getaffinity(0, sizeof old_, &old_) != 0 || !CPU_ISSET(cpu, &old_))
      return;
    CPU_ZERO(&one);
    CPU_SET(cpu, &one);
    pinned_ = ::sched_setaffinity(0, sizeof one, &one) == 0;
#endif
  }
  ~ThreadOnCpu() {
#ifdef _WIN32
    if (old_) ::SetThreadAffinityMask(::GetCurrentThread(), old_);
#else
    if (pinned_) ::sched_setaffinity(0, sizeof old_, &old_);
#endif
  }
  ThreadOnCpu(const ThreadOnCpu&) = delete;
  ThreadOnCpu& operator=(const ThreadOnCpu&) = delete;

  bool pinned() const {
#ifdef _WIN32
    return old_ != 0;
#else
    return pinned_;
#endif
  }

 private:
#ifdef _WIN32
  DWORD_PTR old_ = 0;
#else
  cpu_set_t old_;
  bool pinned_ = false;
#endif
};

// A file opened for reading at offsets from several threads.
class File {
 public:
  explicit File(const std::string& path) {
#ifdef _WIN32
    h_ = ::CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                       FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h_ == INVALID_HANDLE_VALUE)
      throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "cannot open '" + path + "'");
#else
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "cannot open '" + path + "'");
#endif
  }
  ~File() {
#ifdef _WIN32
    ::CloseHandle(h_);
#else
    ::close(fd_);
#endif
  }
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  // Hint that [offset, offset + n) is read front to back (Linux; Windows opens the file with the same hint).
  void advise_sequential([[maybe_unused]] uint64_t offset, [[maybe_unused]] uint64_t n) const {
#ifndef _WIN32
    ::posix_fadvise(fd_, static_cast<off_t>(offset), static_cast<off_t>(n), POSIX_FADV_SEQUENTIAL);
#endif
  }

  // Up to n bytes at `offset` (fewer at the end of the file; 0 past it). Throws std::system_error.
  size_t read_at(void* buf, size_t n, uint64_t offset) const {
#ifdef _WIN32
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(offset);
    ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
    DWORD got = 0;
    if (!::ReadFile(h_, buf, static_cast<DWORD>(std::min<size_t>(n, 1u << 30)), &got, &ov)) {
      const DWORD err = ::GetLastError();
      if (err == ERROR_HANDLE_EOF) return 0;
      throw std::system_error(static_cast<int>(err), std::system_category(), "read");
    }
    return got;
#else
    for (;;) {
      ssize_t r = ::pread(fd_, buf, n, static_cast<off_t>(offset));
      if (r >= 0) return static_cast<size_t>(r);
      if (errno != EINTR) throw std::system_error(errno, std::generic_category(), "read");
    }
#endif
  }

 private:
#ifdef _WIN32
  HANDLE h_ = INVALID_HANDLE_VALUE;
#else
  int fd_ = -1;
#endif
};

}  // namespace artlite::platform
