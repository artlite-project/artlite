"""The prefill attention's and norm's variants on the host: the validated modes and the designs they select."""

import pytest

from artlite.prefill.attn import AttnMode, Kind, Shape
from artlite.prefill.attn.layout import core_bytes
from artlite.prefill.norm import Design, PrefillNormOperator

LLAMA = Shape(hd=128, qh=3, kp=1, a_bytes=9280)  # (a Llama-like shape: single K)
HILO = Shape(hd=128, qh=3, kp=2, a_bytes=9280)
LFM = Shape(hd=64, qh=2, kp=1, a_bytes=9280, per=2, gate=True)


def test_attn_mode_kinds() -> None:
    assert AttnMode.make(LLAMA, T=512, m=1).kind is Kind.CAUSAL
    packed = AttnMode.make(LLAMA, T=512, m=1, window=2)
    assert packed.kind is Kind.PACKED and packed.has_seg and packed.window == 2
    assert AttnMode.make(LLAMA, T=512, m=1, swa=LLAMA.pg * 4, kv0=LLAMA.group * 4).kind is Kind.SWA
    nc = AttnMode.make(LFM, T=512, m=1, causal=False)
    assert nc.kind is Kind.NONCAUSAL and nc.has_seg and nc.bcast and not AttnMode.make(LFM, T=512, m=1, bcast=False,
                                                                                        causal=False).bcast
    assert not AttnMode.make(LLAMA, T=512, m=1).has_seg and not AttnMode.make(LLAMA, T=512, m=1, bcast=True).bcast


def test_attn_mode_refuses_combinations() -> None:
    with pytest.raises(AssertionError):  # (the non-causal design is alone)
        AttnMode.make(LLAMA, T=512, m=1, causal=False, window=2)
    with pytest.raises(AssertionError):  # (a sliding window: whole cache groups, with its guard)
        AttnMode.make(LLAMA, T=512, m=1, swa=LLAMA.pg + 8)
    with pytest.raises(AssertionError):  # (the ring: hi + lo K and a pass per group)
        AttnMode.make(LLAMA, T=512, m=1, window=2, ring=3)
    with pytest.raises(AssertionError):  # (the ring needs the packed window)
        AttnMode.make(HILO, T=512, m=1, ring=3)
    assert AttnMode.make(HILO, T=512, m=1, window=2, ring=3).kind is Kind.RING
    with pytest.raises(AssertionError):  # (the broadcast: LFM3's shapes with gates)
        AttnMode.make(Shape(hd=64, qh=2, kp=1, a_bytes=9280, per=2), T=512, m=1)


def test_blocks_fit_the_core() -> None:
    for s in (LLAMA, HILO, LFM):
        qb, cr = s.blocks
        assert (qb, cr) in ((2, 4), (1, 4)) and core_bytes(s, qb, cr) <= 0x10000 - 512


@pytest.mark.parametrize("kw,design", [({"E": 2048}, Design.PLAIN), ({"E": 2048, "add": False}, Design.PLAIN),
                                       ({"E": 4096}, Design.REPLAY), ({"E": 2048, "deltas": 2}, Design.REPLAY),
                                       ({"E": 2048, "deltas": 3}, Design.HOLD)])
def test_norm_design(kw: dict, design: Design) -> None:
    op = PrefillNormOperator(T=64, **kw)
    assert op.design is design and op.hold == (design is Design.HOLD) and op.replay == (design is Design.REPLAY)
