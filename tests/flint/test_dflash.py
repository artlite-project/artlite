"""Flint's DFlash2 drafter (``artlite.decode.dflash``, incoai's Flint-DFlash2 Q4_K_M) on the NPU, the whole model:

- greedy text the same with and without the drafter (the model verifies every draft), on a code prompt; the drafts'
  acceptance (llama.cpp's on the iGPU at 3 drafts a round: 0.66 on code; the NPU's 0.81 over these 200 tokens, its
  rounds taking the drafts that pay);
- the drafter's rows of a prompt written by the prefill (``prefill.dflash``) against the injections' of the same prompt
  run as decode steps: the drafter's next block's logits within the two paths' rounding (5.2%: the taps are the model's
  residuals, a few % apart between its prefill and its decode steps, test_prefill.py), the same drafts.
"""

import numpy as np
import pytest

from artlite.gguf_model import GGUFFile
from artlite.model_config import flint_dflash_of, flint_gguf
from models.flint.flint import Flint, NpuFlint
from models.sampling import Sampling

from ..model_files import require
from ..numerics import rel

pytestmark = pytest.mark.slow  # the whole model: run with --run-slow

CODE = ("Write a Python function that checks whether a string is a palindrome, ignoring case and non-alphanumeric "
        "characters.")


def _files() -> tuple[str, str]:
    path = flint_gguf()
    require(path)
    dflash = flint_dflash_of(path)
    if dflash is None:
        pytest.skip(f"no DFlash2 drafter of {path}")
    return path, dflash


def test_flint_dflash_text() -> None:
    path, _ = _files()
    out = {}
    for spec in ("draft-dflash", "none"):
        f = Flint(path, spec_type=spec)
        m = f.load(capacity=2048)
        ids = f.chat([{"role": "user", "content": CODE}])
        m.lookup = False  # (the drafter's rounds alone)
        logits = m.forward(ids)
        m.prepare(len(ids))  # (after a dispatch: the drafter's context opens, the model's code pages load again)
        d0, a0 = m.drafted, m.accepted
        out[spec] = list(m.generate(logits, 200, Sampling(greedy=True), ids))
        if spec != "none":
            drafted, accepted = m.drafted - d0, m.accepted - a0
            print(f"{drafted} drafts, {accepted} accepted ({accepted / drafted:.2f}), {m.blocks} blocks")
            assert drafted and accepted / drafted > 0.45
        del m
    assert out["draft-dflash"] == out["none"]


def test_flint_dflash_prefill() -> None:
    """(a prompt of 600 tokens: a chunk of 512 and decode steps after it; the same as decode steps alone)"""
    path, dflash = _files()
    f = Flint(path, spec_type="none")
    ids = f.chat([{"role": "user", "content": CODE + " Explain each step. " * 60}])[:600]
    rows = {}
    for prefill in (True, False):
        m = NpuFlint(GGUFFile(path), capacity=2048, dflash=dflash, prefill=prefill)
        m.reset()
        logits = m.forward(ids)
        token = int(np.argmax(logits))
        d = m.flash.block(token, 3)  # (the injection of the last step's rows and the block at the next position)
        rows[prefill] = (np.asarray(m.dflash.logits(), np.float32).copy(), d)
        del m
    err = rel(rows[True][0], rows[False][0])
    print(f"block logits rel {err:.4f}; drafts {rows[True][1]} / {rows[False][1]}")
    assert err < 0.08 and rows[True][1] == rows[False][1]
