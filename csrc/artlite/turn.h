#pragma once

#include <cstddef>
#include <cstdint>

// The NPU's turn between processes. Hardware contexts of different processes share the array: the firmware runs their
// commands in an order of its own, and a switch to another context loses what a design keeps on the array beyond its
// configuration (code pages' stores: artlite.overlay.OverlayDevice.set_prologue). A process holds the turn while any
// of its commands is pending (Kernel's submits take it, the waits that retire them give it back) and while a caller
// holds it (enter / leave: a check of foreign() and the submits it decides), so no other process's command runs
// between two of its own meanwhile; foreign() counts the turns this process took after another process had one.
//
// A lock file every user's processes open (Linux: /dev/shm, else the temp directory; Windows: the temp directory):
// byte 0 is the turn, a process waiting for it holds byte 1 shared (contended), bytes 8-15 hold the last holder's id,
// and each process holds one of bytes 64-319 from its first call on (peers: whether another holds one; a range lock
// tests them all at once). Where the file cannot be opened there is no turn (every call returns at once, foreign()
// stays 0, peers() false).
namespace artlite::turn {

// Until this process holds the turn (nested holds count; the commands' are among them).
void enter();
// Give back n holds; the last one gives back the turn.
void leave(size_t n = 1);
// The turns this process took after another process's (in this process's lifetime).
uint64_t foreign();
// Seconds this process waited for another process's turn to end (in its lifetime): a timing of this process's own
// work subtracts it.
double waited();
// Whether another process waits for the turn.
bool contended();
// Whether another process uses the NPU: one that called any of these and has not ended (its turns may come between
// this process's dispatches at any time).
bool peers();
// Holding nothing: until a waiting process has taken the turn (at most kPassMs), so that the next enter() comes after
// its turn. Holding something: returns at once.
void pass();

constexpr int kPassMs = 50;

}  // namespace artlite::turn
