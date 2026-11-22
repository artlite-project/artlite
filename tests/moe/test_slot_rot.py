"""``moe.slot_rot.SlotRotOperator`` on the NPU: the experts' h panels in their chunk slots rotated in place, each chunk's
items (32 tokens) over its valid rows by the downs' rotation (Qwen3.6-35B-A3B-PARO's, or a random one), bit for bit
against ``prefill.rotate``'s host model; the slots' rows past the items untouched."""

import numpy as np

from artlite.moe.expert_gemm import CHUNK, N_ITEMS, N_PARAM
from artlite.moe.slot_rot import TOK, SlotRotOperator
from artlite.operators import bfp16
from artlite.operators.design import COLS
from artlite.paro import Rotation
from artlite.prefill.norm import PrefillNormOperator

from ..prefill.test_rotate import model


def _rotation(rng: np.random.Generator) -> Rotation:
    try:
        from artlite.hf_model import HfModel
        from artlite.model_config import checkpoint

        return HfModel(checkpoint("qwen3.6-35B-A3B-paro")).tensor("blk.1.ffn_down_exps.weight").rot()
    except (OSError, FileNotFoundError, KeyError):
        pairs = np.stack([np.concatenate([rng.permutation(128) for _ in range(4)]) for _ in range(8)])
        return Rotation(rng.uniform(-1.2, 1.2, (8, 256)), pairs, rng.uniform(-2, 2, 512), 128)


def test_slot_rot() -> None:
    import artlite
    from artlite.operators.fused import Library
    from artlite.txn import device_address

    rng = np.random.default_rng(3)
    K, cap = 512, 6
    tf = _rotation(rng).tan_form(None, unit=True)
    op = SlotRotOperator(K=K, cap=cap)
    lib = Library("t_slot_rot", [op, PrefillNormOperator(T=128, E=1024, eps=1e-6)]).compile().load()  # (a library: two)
    dev = lib.device
    sizes = [rng.integers(1, CHUNK + 1, rng.integers(1, cap + 1)) for _ in range(COLS)]  # (a column's chunks' rows)
    x = (rng.standard_normal((COLS, cap, CHUNK, K)) * np.exp(rng.standard_normal((COLS, cap, CHUNK, 1)))).astype(np.float32)
    hp = np.stack([np.stack([bfp16.to_panels(x[c, k]) for k in range(cap)]) for c in range(COLS)])  # ([c][k][panels])
    h = artlite.from_numpy(hp.reshape(-1).copy(), dev)
    cb = artlite.from_numpy(np.zeros(COLS * op.cb_words * 2, bfp16.bfloat16), dev)
    ro = artlite.from_numpy(tf.pair_objects(TOK).reshape(-1).view(bfp16.bfloat16), dev)
    a = {n: device_address(b, 0, lib._ctx) for n, b in (("H", h), ("CB", cb), ("RO", ro))}
    v = cb._raw.map().view(np.uint32)
    op.streams(sizes, cb=[a["CB"] + 4 * c * op.cb_words for c in range(COLS)],
               h=[a["H"] + 2 * c * cap * op.slot for c in range(COLS)],
               out=[v[c * op.cb_words:(c + 1) * op.cb_words] for c in range(COLS)])
    cb._raw.sync_to_device()
    lib.set_params({f"{N_PARAM}{c}": n << N_ITEMS for c, n in enumerate(op.items(sizes))})
    lib.run([(0, [a["CB"], a["RO"], a["H"], a["H"]])], bufs=[h._raw, cb._raw, ro._raw])
    h.sync_from_device()
    got = h.numpy().reshape(COLS, cap, -1)
    bad = 0
    for c in range(COLS):
        for k in range(cap):
            vals = bfp16.from_panels(hp[c, k], CHUNK, K)
            g = -(-int(sizes[c][k]) // TOK) if k < len(sizes[c]) else 0
            want = vals.copy()
            if g:
                rows = TOK * g
                want[:rows] = bfp16.from_panels(bfp16.to_panels(model(vals[:rows], tf)), rows, K)
            bad += int((bfp16.from_panels(got[c, k], CHUNK, K) != want).sum())
    assert bad == 0, f"{bad} values differ"
