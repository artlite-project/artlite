"""What the decode overlays' attention cores share so that a token slot's attention depends on its own position and
the positions before it only - not on the other slots of its step, nor on what a rolled-back step wrote (the kernel
side: ``kernel_lib.h``'s V records and ``al::merge``):

- sub-steps (``substeps``): a layer's attention runs the slots of one cache group at a time (at most as many as the
  cores hold query rows for), so that the cores' split of the cache follows from the slot's group alone;
- the open V records' exact values: a V record of 8 positions is per dim one block over the positions, and the
  values of an open one (whose positions the steps still add) stay exact in a ring next to the caches - per (layer,
  KV head) two slots, record r in slot r & 1, its rows [d-block][position 8][8] bf16 (``ring_rows``) - from which the
  cores encode its panels up to the position that reads them (``VRing``: which records the rings hold).
"""

import numpy as np
import torch


def substeps(p: int, S: int, pg: int, r: int) -> list[tuple[int, int]]:
    """The sub-steps (first slot, slots) of a layer's attention in a step at positions p .. p + S - 1: the slots of one
    cache group of pg positions, at most r of them."""
    out, a = [], 0
    while a < S:
        n = min(S - a, r, pg - (p + a) % pg)
        out.append((a, n))
        a += n
    return out


def tails(S: int) -> int:
    """Tail objects of a (sub-)step of S slots (the ring slots of its first and last records; with one record the same
    twice: the same transfers for every step of S; one slot has one record)."""
    return 1 if S == 1 else 2


def ring_rows(slot: np.ndarray, hd: int) -> np.ndarray:
    """A ring slot (bytes: [d-block][position 8][8] bf16) -> its record's rows (8, hd) float32."""
    v = torch.from_numpy(np.ascontiguousarray(slot[:16 * hd])).view(torch.bfloat16).float().numpy()
    return np.ascontiguousarray(v.reshape(hd // 8, 8, 8).transpose(1, 0, 2)).reshape(8, hd)


class VRing:
    """Which open records the rings of an overlay's layers hold: per layer the (record, rows up to (exclusive)) of each
    slot as the last steps left them; and the layers whose cache a prefill wrote last (it writes no ring: their open
    record's values are its V panels, exact as bf16 rows - a step reads them instead: record mode)."""

    def __init__(self) -> None:
        self.slots: dict[int, dict[int, tuple[int, int]]] = {}
        self.from_prefill: set[int] = set()

    def clear(self) -> None:
        self.slots.clear()
        self.from_prefill.clear()

    def check(self, layer: int, p: int, end: int) -> None:
        """A step at p reads the rows of record p >> 3 up to ``end`` (exclusive) from the ring of ``layer``: the ring
        holds them (a step continues from the last one's positions, at most its own back, as a generation loop's
        rejected drafts do)."""
        if p % 8 == 0 and end <= p:
            return
        r, e = self.slots.get(layer, {}).get((p >> 3) & 1, (-1, 0))
        assert r == p >> 3 and e >= end, f"layer {layer}: the ring holds rows of record {r} to {e}, not {p >> 3} to {end}"

    def ran(self, layer: int, p: int, S: int) -> None:
        """A step of S slots at p wrote the rows of its records into the ring of ``layer``."""
        ring = self.slots.setdefault(layer, {})
        r0, r1 = p >> 3, (p + S - 1) >> 3
        ring[r0 & 1] = (r0, min(p + S, 8 * r0 + 8))
        if r1 != r0:
            ring[r1 & 1] = (r1, p + S)
        self.from_prefill.discard(layer)
