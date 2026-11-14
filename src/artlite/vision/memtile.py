"""First-fit allocation of the vision overlay's memtile bytes (the same addresses in every column)."""

from . import design as D


class Mem:
    """First-fit allocation of the columns' memtile bytes [0, size) (64-byte aligned)."""

    def __init__(self, size: int = D.HEAP):
        self.free = [(0, size)]
        self.size = size

    def alloc(self, n: int, *, top: bool = False) -> int:
        """n bytes from the first free range that holds them (top: the end of the last one: small buffers that live
        across ops, away from the large tensors)."""
        n = -(-n // 64) * 64
        for i in (reversed(range(len(self.free))) if top else range(len(self.free))):
            a, s = self.free[i]
            if s >= n:
                if top:
                    self.free[i] = (a, s - n)
                    a += s - n
                else:
                    self.free[i] = (a + n, s - n)
                if self.free[i][1] == 0:
                    del self.free[i]
                return a
        raise MemoryError(f"memtile: {n} B not free ({self.free})")

    def release(self, a: int, n: int) -> None:
        """Give back the n bytes from ``a`` (as ``alloc`` took them)."""
        n = -(-n // 64) * 64
        self.free.append((a, n))
        self.free.sort()
        merged = []
        for b, s in self.free:
            if merged and merged[-1][0] + merged[-1][1] == b:
                merged[-1] = (merged[-1][0], merged[-1][1] + s)
            else:
                merged.append((b, s))
        self.free = merged
