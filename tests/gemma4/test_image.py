"""The Gemma 4 weight image of a PARO checkpoint (models/gemma4/image.py), on the host (no NPU): z-lab's E4B PARO
layout (KC 1024: P4_128 tiles, the hidden's inputs as half regions, gate / up and the LM head's plane in pairs of row
tiles; the R phases' rotation objects of each column's groups with w_post in their spare bytes, rot_o / rot_down /
rot_proj's) and the packed tiles at their places (``image_checks.check_tails`` against the padded image)."""

import numpy as np
import pytest
from ml_dtypes import bfloat16

from artlite.decode.design import ROT_GROUP as ROT
from artlite.decode.design import RWP_AT, RWS_AT
from artlite.decode.protocol import rot_objects
from artlite.decode.screen import derived
from artlite.hf_model import HfModel
from artlite.model_config import checkpoint
from artlite.operators.design import COLS
from artlite.paro import ParoTensor, Rotation
from models.gemma4.image import KC_PARO, PARO_TAILS, PLE_Q8, R_GROUPS, GemmaImage

from ..decode.image_checks import check_tails
from ..model_files import require

E4B = "gemma-4-E4B-paro"


@pytest.fixture(scope="module")
def hf() -> HfModel:
    require(checkpoint(E4B))
    return HfModel(checkpoint(E4B))


def test_paro_layout(hf: HfModel) -> None:
    """Layers 0 (own KV, sliding), 5 (own KV, full) and 24 (KV-shared): every phase's region in step order, the rows of
    q / k / v (k and v: a KV head's quarters), the R phases' objects (each column's groups' rotation objects with the
    norm weight folded in, a padding object of zeros past 2 groups, w_post of the groups and of the residual slice),
    rot_o / rot_down / rot_proj's groups (rot_proj: proj's group c % 2 on column c), the blocks in pairs, the LM head's
    Q4_0 plane."""
    img = GemmaImage(hf, layers=[0, 5, 24], vocab=COLS * 64)
    cfg = img.cfg
    H, h8 = cfg.hidden_size, cfg.hidden_size // COLS
    assert img.paro and img.kc == KC_PARO and img.screen and img.formats() == {"P4_128", "Q4_0", "Q8_0"}
    tiles = ["gate", "up", "rot_down", "down", "rpgate", "pgate", "rot_proj", "proj"]
    own = ["rq", "rk", "rv", "q", "k", "v", "rot_o", "o", "rgate", "rup"]
    want = [(-1, "ple")] + [(L, n) for L in (0, 5) for n in own + tiles]
    want += [(24, n) for n in ["rq", "q", "rot_o", "o", "rgate", "rup"] + tiles] + [(-1, "rlm"), (-1, "lm")]
    assert [(r.layer, r.name) for r in img.regions] == want
    assert {(r.name, b.name) for r in img.regions for b in r.blocks if b.pt} == {("gate", "gate"), ("up", "up"),
                                                                                ("lm", "lm")}
    assert all(r.half == (r.name in PARO_TAILS) for r in img.regions if not r.data)
    hd = cfg.head_dim(5)
    assert img.region(5, "q").block("q").rows[3].tolist() == list(range(3 * hd, 4 * hd))
    k = img.region(5, "k").block("k").rows
    assert k[6].tolist() == list(range(hd + 2 * hd // 4, hd + 3 * hd // 4))  # (KV head 1's quarter 2)
    assert img.region(-1, "lm").block("lm").fmt == "Q4_0" and derived("Q4_0", img.qtype("token_embd.weight"))
    # an R phase's objects: per column its groups (3, 3, 3, 3, 2, 2, 2, 2), two an object
    T = img.tile_bytes
    post = np.asarray(hf.tensor("blk.4.post_norm.weight").dequantize(), np.float32).reshape(-1)  # (layer 5's after 4)
    img5 = GemmaImage(hf, layers=[4, 5], vocab=COLS * 64)
    objs = img5.data_objects(5, "rk")
    gamma = np.asarray(hf.tensor("blk.5.attn_norm.weight").dequantize(), np.float32).reshape(-1)
    tan = hf.tensor("blk.5.attn_k.weight").rot().tan_form(gamma, unit=True).objects()
    assert objs.shape == (COLS, 2, T) and img5.region(5, "rk").data == 2 * T
    for c in range(COLS):
        g0, ng = img5.r_groups(c)
        assert ng == R_GROUPS[c] and g0 == sum(R_GROUPS[:c])
        rot = rot_objects(np.concatenate([tan[g0:g0 + ng], np.zeros((-ng % 2, tan.shape[1]), np.uint8)]), T)
        for o in range(img5.r_objects(c)):
            assert np.array_equal(objs[c, o, :RWP_AT], rot[o, :RWP_AT]), (c, o)
            gs = slice(ROT * (g0 + 2 * o), ROT * min(g0 + 2 * o + 2, g0 + ng))
            n = 2 * (gs.stop - gs.start)
            assert np.array_equal(objs[c, o, RWP_AT:RWP_AT + n].view(bfloat16), post[gs].astype(bfloat16)), (c, o)
        assert np.array_equal(objs[c, 0, RWS_AT:RWS_AT + 2 * h8].view(bfloat16),
                              post[h8 * c:h8 * (c + 1)].astype(bfloat16))
        if img5.r_objects(c) < 2:
            assert not objs[c, 1].any()
    assert not img.data_objects(0, "rq")[:, :, RWP_AT:RWS_AT + 2 * h8].any()  # (layer 0: no update, w_post zero)
    # the identity copy of the LM head over the output norm
    w = np.asarray(hf.tensor("output_norm.weight").dequantize(), np.float32).reshape(-1)
    ident = Rotation.identity(H).tan_form(w, unit=True).objects()
    assert np.array_equal(img.data_objects(-1, "rlm")[0, 0, :RWP_AT], rot_objects(ident[:2], T)[0, :RWP_AT])
    pt = hf.tensor("blk.24.proj.weight").rot().tan_form(unit=True).objects()
    rp = img.data_objects(24, "rot_proj")
    assert rp.shape == (COLS, 1, T)
    for c in range(COLS):  # (column c: group c % 2, for copy c // 2 of proj's x)
        assert np.array_equal(rp[c, 0], rot_objects(pt[c % 2:c % 2 + 1], T)[0])
    assert img.region(24, "rot_o").data == T and img.region(5, "rot_o").data == 2 * T  # (head_dim 256, 512)
    assert img.region(24, "rot_down").data == 5 * T  # (a column's 1280 channels)


def test_paro_pack(hf: HfModel) -> None:
    """The packed image of layers 0 and 24 (a slice of the LM head): the half regions' tiles and pairs as the padded
    image's rows pack (``check_tails``: the PARO linears as their rows of blocks, the head's Q4_0 plane, the per-layer
    projection's Q8_0 copy)."""
    img = GemmaImage(hf, layers=[0, 24], vocab=COLS * 64 * 3)
    pad = GemmaImage(hf, layers=[0, 24], vocab=COLS * 64 * 3, tails=False)

    def rows_of(name: str) -> object:
        if name == PLE_Q8:  # (its copy, dropped after the pack)
            img._ple()
        t = img.tensor(name)
        return img.paro_rows(name) if isinstance(t, ParoTensor) or derived("Q4_0", t.qtype) else t

    check_tails(img, pad, set(PARO_TAILS), tensor=rows_of)
