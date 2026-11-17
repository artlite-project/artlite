"""The attention's variants as one validated mode (``AttnOperator(mode)``): which keys a row sees and where the cores
stream them from."""

import enum
from dataclasses import dataclass

from .shape import Shape


def swa_groups(swa: int, pg: int) -> int:
    """The cache groups before a position's that a sliding window of ``swa`` positions reaches (its keys p - swa + 1 ..
    p: from the window's first group, ``pg`` positions a group)."""
    return -(-(swa - 1) // pg)


class Kind(enum.Enum):
    CAUSAL = "causal"  # every row its chunk's keys up to itself and the history before the chunk
    PACKED = "packed"  # several inputs (segments) in the chunk: each row its segment's keys up to itself
    RING = "ring"  # the packed variant with the memory tiles holding the history and a ring of the window's groups
    SWA = "swa"  # a sliding window of positions (a cache with a guard before its group 0)
    NONCAUSAL = "noncausal"  # a prompt in one chunk at position 0: every row the state's keys and its question's


@dataclass(frozen=True)
class AttnMode:
    """``window``: the packed variant's window of cache groups (``Kind.PACKED``, ``RING``); ``ring``: the history's
    groups held in the memory tiles (``RING``); ``swa``: the sliding window's positions, whole cache groups or one
    position more (Flint's 513: keys p - 512 .. p), and ``kv0``: the bytes of a KV head's cache before its group 0
    (``SWA``); ``bcast``: two columns per KV head (``Shape.per``) share its streams, read once by column 2 h's shim and
    broadcast by its memtile."""

    kind: Kind
    window: int = 0
    ring: int = 0
    swa: int = 0
    kv0: int = 0
    bcast: bool = False

    @staticmethod
    def make(shape: Shape, *, T: int, m: int, window: int = 0, ring: int = 0, swa: int = 0, kv0: int = 0,
             bcast: bool = True, causal: bool = True) -> "AttnMode":
        """The mode of the options of a chunk of T tokens (``m``: passes per cache group), refused where they do not
        combine: the non-causal design is alone (full layers, int16 keys), the broadcast is the causal and sliding-window
        designs', a sliding window is whole cache groups (or one position more) with its guard (``swa_groups``), the ring
        is hi + lo K with a pass per group."""
        s = shape
        assert window >= 0
        bc = s.per == 2 and bcast
        assert causal or (not window and not ring and not swa and T < 32768), "non-causal: full layers, int16 keys"
        assert not bc or (not window and not ring), "the broadcast: the causal and sliding-window designs"
        assert not swa or (swa % 8 in (0, 1) and (swa - swa % 8) % s.pg == 0 and not window and not ring
                           and kv0 >= swa_groups(swa, s.pg) * s.group), swa
        if ring:
            assert window and m == 1 and s.kp == 2 and s.used % (2 * s.unit) == 0, "the ring: hi + lo K, a pass a group"
        kind = (Kind.NONCAUSAL if not causal else Kind.SWA if swa else Kind.RING if ring else
                Kind.PACKED if window else Kind.CAUSAL)
        return AttnMode(kind, window, ring, swa, kv0, bc)

    @property
    def has_seg(self) -> bool:
        """Whether the runtime sequence has the SEG argument (each pass's rows' first keys)."""
        return self.kind in (Kind.PACKED, Kind.RING, Kind.NONCAUSAL)
