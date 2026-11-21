"""The rounds of a ParoQuant rotation in its tan form (``artlite.paro.TanForm``), the PARO linears' input transform
(template: ``rotate.rounds.cc.jinja``): per round every storage position s of a group of 128 channels takes
``u[s] += T[r, s] u[partner[r, s]]`` (the old values), each channel holding 4 lanes (a design's token slots, a
prefill's 4 tokens). One numeric form (paro's research: KL 1.6e-4 against the exact transform on Qwen3-0.6B-PARO, 0.3%
of the quantization's own 5.3e-2): u in fp32 (the self term exact), the partner's bf16 copy times T in bf16 on the fp32
MAC, the copy of the round's result written into the other of two bf16 buffers (laid out as the 4x64 gather reads: each 4 channels'
[4] bf16 twice in 64 bytes, 2 KB a group).

The coefficients of a group as the NPU reads them (``coefs``): per round [T: 128 bf16 | partner: 128 u8] (384 B), 8
rounds 3 KB; the caller initializes u = x w' (fp32) and its bf16 copy in b0 (``copy_layout``), and reads u after the call
(the rounds' storage order: ``TanForm.chan``).

Schedule (measure_ii, a round of one group): II 17 per 16 channels, 179 cycles a round, ~1550 per group of 8 rounds
(both state widths), the stores binding (the copy's 4-channel groups written twice: 8 32-byte stores and 4 of u per 16
channels). A round is a call of its own: llvm-aie gives a gather a memory operand only when each base of its address
vector is a pointer parameter used for nothing else (``kmath.gather``; then it takes the gathers as reads of a read-only
table, so the call orders a round's gathers after the stores of the round before). Rejected: rounds inline with their
gathers issued blocks ahead (II 16-29: the copies of the prefetched registers bind the move slot), several groups per
call (II 22-24: the per-group base), a copy of each 64-bit entry once (the gather ignores address bits 3:0 and reads its
odd lanes' entries 32 bytes on: only the doubled layout reads any channel from any lane).
"""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

from .fn import Fn, make

GROUP = 128  # channels of a rotation group
ROUND_BYTES = 3 * GROUP  # a round's coefficients: T bf16, then the partners u8
LANES4 = 4  # lanes per channel


def rotate(rounds: int = 8, *, groups: int = 1, f32: bool = True, cstride: int = 0, inline: bool = False) -> Fn:
    """The ``rounds`` rounds of ``groups`` groups at once (module docstring): a function ``([u,] ga0, gb0, ga1, gb1, b0,
    b1, coef)``: ``f32``: u fp32 [group][128][4] the state (b0, b1 its bf16 copies), else the state is the bf16 copy
    itself (b0 at entry; KL 6.1e-4 against 1.6e-4: paro's research); ga0 = gb0 = b0 and ga1 = gb1 = b1 the copies
    again (each copy's gather bases: llvm-aie's memory operands, the template's comment); the coefficients of group g at
    coef + ``cstride`` g (default: rounds x 384 B, the groups' back to back). The result: u (f32), or the last round's
    copy (b0 for an even count of rounds, else b1)."""
    if rounds < 1 or groups < 1:
        raise ValueError(f"kmath: rotate rounds = {rounds}, groups = {groups}")
    cs = cstride or rounds * ROUND_BYTES
    return make(f"rot_r{rounds}_g{groups}_{'f32' if f32 else 'bf16'}_c{cs}" + ("_i" if inline else ""),
                "rotate.rounds.cc.jinja", inline=inline, lanes=32, arg="ptr", ret="ptr", func="rotate",
                params={"rounds": rounds, "groups": groups, "f32": f32, "cstride": cs},
                variables={"rounds": rounds, "RB": ROUND_BYTES, "groups": groups, "f32": f32, "cstride": cs})


def coefs(T: np.ndarray, partner: np.ndarray) -> np.ndarray:
    """A group's coefficients (rounds, 128) T (fp64) and partners -> the bytes the kernel reads (rounds x 384 B)."""
    R = T.shape[0]
    out = np.zeros((R, ROUND_BYTES), np.uint8)
    out[:, :2 * GROUP] = np.asarray(T, np.float32).astype(bfloat16).view(np.uint8).reshape(R, -1)
    out[:, 2 * GROUP:] = np.asarray(partner, np.uint8)
    return out.reshape(-1)


def copy_layout(u: np.ndarray) -> np.ndarray:
    """A group's values (128, 4) -> its copy as the kernel's buffers hold it (32 groups of 4 channels, each twice)."""
    v = np.asarray(u).reshape(GROUP // 4, 1, 16)
    return np.concatenate([v, v], axis=1).reshape(-1)
