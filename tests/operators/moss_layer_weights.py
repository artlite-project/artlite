"""The encoder layer's weight images (the overlay's layouts) packed from the HF checkpoint, with host tests.

The layer program (``models.moss_tr.chain``) reads one bf16 image per matrix in the *vision overlay's* operand
layouts, so this module owns that packing and nothing else:

- **B (weights)**: ``vision``'s ``Gemm`` weights: per pass of 80 output channels its k-tiles, each
  ``[16-column block][k-step][half][8 outputs x 8 k]`` bfp16 panels (72 B) — the MAC loop's pop order
  (``layout.PANEL``, ``weights.Slots``). ``N`` is a multiple of 320 and of 80 per pass; ``K`` a multiple of the k-tile
  64 (the residual width 1280 = 20 k-tiles).
- **bias as an extra K channel**: the layer's GEMMs have no bias (Whisper's projections are bias-free except v / o;
  those are folded the same way as the stem's: A's K-th channel is 1.0, W's K-th column the bias) — ``fold_bias``.
- **LN / pos vectors**: fp32 gamma/beta (the ``Apply`` op's B slots read fp32), and the position table as bf16 rows in
  the C-tile layout (added by the ``Gemm``'s ``res`` C tiles).
- **KV slots**: the attention's K / V^T image for the round's 8 heads (``layout.kv_slots``: d 80, groups of 64 keys).
"""

import numpy as np

from artlite.vision import layout as L

KT = 64  # the overlay's k-tile (vision/design.py)
PASS = 80  # output channels of a core's pass (the overlay's Gemm geometry)
OUT_PANEL = 16  # output channels of a B block


def b_image(w: np.ndarray) -> np.ndarray:
    """(N, K) -> the overlay's B image (bfp16 panels, the MAC's pop order): [pass][k-tile][16-col block][k-step]
    [half][8 outputs x 8 k]."""
    n, kk = w.shape
    assert n % PASS == 0 and kk % KT == 0 and n % (2 * OUT_PANEL) == 0, (n, kk)
    p = L.panels(w).reshape(kk // 8, n // 8, L.PANEL)            # [k-block][8-output panel][72 B]
    npass, nkt, ks = n // PASS, kk // KT, KT // 8
    v = p.reshape(nkt, ks, npass, PASS // OUT_PANEL, 2, L.PANEL)  # [kt][k-step][pass][block][half][72]
    return np.ascontiguousarray(v.transpose(2, 0, 3, 1, 4, 5)).reshape(-1)


def b_decode(img: np.ndarray, n: int, kk: int) -> np.ndarray:
    """The B image back to (N, K) fp32 (the test's inverse; the panels' blocks are 72 B)."""
    npass, nkt, ks = n // PASS, kk // KT, KT // 8
    v = np.asarray(img).view(np.uint8).reshape(npass, nkt, PASS // OUT_PANEL, ks, 2, 8, 9)
    v = L.unblocks(v)                                            # [pass][kt][block][k-step][half][8 outputs][8 k]
    w = np.zeros((npass * PASS, nkt * KT), np.float32)
    for b in range(nkt):
        for p in range(npass):
            for blk in range(PASS // OUT_PANEL):
                for st in range(ks):
                    for h in range(2):
                        o0 = p * PASS + blk * OUT_PANEL + h * 8      # (a block's two halves are 8 outputs each)
                        w[o0:o0 + 8, b * KT + st * 8:b * KT + (st + 1) * 8] = v[p, b, blk, st, h]
    return np.ascontiguousarray(w)


def fold_bias(w: np.ndarray, bias: np.ndarray | None, *, kpad: int, npad: int) -> np.ndarray:
    """A projection's weights (N, K_real) -> (npad, kpad): the bias in the K_real-th column (A's 1.0 channel), zero
    rows past N and zero columns past K_real + 1 (the pad keeps the overlay's k-tile and pass geometry)."""
    n, real = w.shape
    assert kpad % KT == 0 and real + 1 <= kpad and npad % 320 == 0 and n <= npad, (n, real, kpad, npad)
    out = np.zeros((npad, kpad), np.float32)
    out[:n, :real] = np.asarray(w, np.float32)
    if bias is not None:
        out[:n, real] = np.asarray(bias, np.float32)
    return out


def ln_vectors(gamma: np.ndarray, beta: np.ndarray | None, pad: int) -> tuple[np.ndarray, np.ndarray]:
    """LayerNorm's gamma / beta (fp32) padded to the residual width: the ``Apply`` op's B slots."""
    g = np.zeros(pad, np.float32)
    g[:gamma.size] = np.asarray(gamma, np.float32)
    b = np.zeros(pad, np.float32)
    if beta is not None:
        b[:beta.size] = np.asarray(beta, np.float32)
    return g, b


def pos_rows(pos: np.ndarray, m: int, n: int) -> np.ndarray:
    """(tokens, C) position table -> the C-tile layout (the ``Gemm``'s ``res`` input); rows past its own are zero."""
    rows = -(-max(m, pos.shape[0]) // 16) * 16
    out = np.zeros((rows, n), np.float32)
    out[:pos.shape[0], :pos.shape[1]] = pos
    return L.ctiles(out, 16)


def kv_slots(k: np.ndarray, v: np.ndarray, *, heads: int = 8) -> np.ndarray:
    """The layer's K / V^T image of the round's ``heads`` heads ((keys, heads, 64) each): the overlay's slot layout
    (``layout.kv_slots``, d 80, groups of 64 keys), fp32 -> the image's bfp16 bytes. The overlay's attention carries
    two images (its CFG batch); MOSS's single sequence uses image 0 and repeats it (harmless, the mask is none)."""
    assert k.shape == v.shape and k.shape[1] == heads, (k.shape, heads)
    assert k.shape[2] == 64, k.shape
    # (the overlay's slots take the heads' dims side by side, each padded to d 80)
    def wide(x: np.ndarray) -> np.ndarray:
        s, h, hd = x.shape
        out = np.zeros((s, h, 80), np.float32)
        out[..., :hd] = x
        return out.reshape(s, h * 80)

    k2 = np.stack([wide(k), wide(k)])          # (2 images, keys, heads * 80)
    v2 = np.stack([wide(v), wide(v)])
    return L.kv_slots(k2, v2, d=80).view(np.uint8)


def npad_of(n: int) -> int:
    """The output channels padded to the overlay's pass geometry (a multiple of 320)."""
    return -(-n // 320) * 320


def layer_images(sd: dict[str, np.ndarray], *, width: int = 1280, ffn: int = 4160) -> dict[str, np.ndarray]:
    """The layer's images from the checkpoint's tensors (HF names, bf16 already as fp32): each matrix folded to ``width``
    (the residual's pad) with its bias in the extra K column, the LN vectors, the position table in the C-tile layout.

    ``sd``: q_w/q_b, k_w, v_w/v_b, o_w/o_b, fc1_w/fc1_b, fc2_w/fc2_b, ln_a_w/ln_a_b, ln_f_w/ln_f_b, pos.
    """
    out: dict[str, np.ndarray] = {}
    for key, w, bias in (("q", sd["q_w"], sd.get("q_b")), ("k", sd["k_w"], None), ("v", sd["v_w"], sd.get("v_b")),
                         ("o", sd["o_w"], sd.get("o_b")), ("fc1", sd["fc1_w"], sd.get("fc1_b")),
                         ("fc2", sd["fc2_w"], sd.get("fc2_b"))):
        # (the K each matrix reads: the residual width, except o (the padded q output) and fc2 (the FFN width))
        kpad = {"q": width, "k": width, "v": width, "o": npad_of(sd["q_w"].shape[0]), "fc1": width, "fc2": ffn}[key]
        wf = fold_bias(w, bias, kpad=kpad, npad=npad_of(w.shape[0]))
        out[key] = b_image(wf)
    for tag in ("ln_a", "ln_f"):
        g, b = ln_vectors(sd[f"{tag}_w"], sd.get(f"{tag}_b"), width)
        out[f"{tag}_g"], out[f"{tag}_b"] = g, b
    out["pos"] = pos_rows(sd["pos"], 1500, width)
    return out


def bf16_decode(img: np.ndarray, tokens: int, c: int) -> np.ndarray:
    """A C-tile image (``layout.ctiles``) back to (tokens, c) fp32 (the test's inverse)."""
    return L.from_ctiles(img, tokens, c, 16)
