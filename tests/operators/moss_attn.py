"""The MOSS encoder's single attention layer as one operator: d 80 (head_dim 64 padded), 16 heads, S keys
= S queries (self-attention, bidirectional, ``valid=None``), exact per-query maximum.

The layer is ``WhisperEncoderLayer``'s self-attention: pre-LN, q/k/v, 16 heads of 64 dims, a full softmax over all S
keys, o. This operator starts after the projections (Q, K, V of one layer, already projected): the host packs them in
the vision overlay's panel layouts and the operator returns the per-head context, one bf16 row per
(query, dim).

Numerics (and ``raise_max``'s alignment with HF): **the reference maximum is the exact per-query maximum over the whole
key set**, computed in a first streaming pass over the keys in fp32 (``p_max``: ``al::max8`` per 8-key group). The second
pass computes P = 2^((s - m) * log2e) with ``kmath``'s fp32 exp2, rounds it **once** to bfp16 panels (the MMUL's operand)
and accumulates o = sum P V and the row sums on the MMUL itself: V's pad dimension 64 is a column of 1.0, so o's dim 64
is the sum of the *same bfp16* P that the numerator took (no separate l bookkeeping), and the finalize divides by it.
Because the maximum is exact and taken over all keys (fp32), P never saturates against a raised reference — the overlay's
``raise_max`` retry does not exist here, which is what HF's softmax does.

Work split: **4 cores, one head each per dispatch** (a dispatch covers 4 of the 16 heads; the host loops 4 dispatches a
layer). The reasons: the shim has 16 S2MM channels (3 input streams a core: Q, K, V -> 8 cores would need 24), and a
memtile holds 48 BD ids (one head x 24 tiles x 2 passes of 24 bands = 1152 K objects = 18 BDs, 576 V objects = 9; with
4 heads a core it is 4x that and aiecc refuses). Per core, per 64-query tile: pass 1 streams the S keys in bands of 64
(K panels, ``MAC<64,64>`` f32), pass 2 streams the keys again (QK, P, then V's panels, ``MAC<64,80>``). Queries and
keys are padded to whole 64-tiles/-bands; the tail band's padded keys get P = 0 (a compile-time mask), the padded
query rows are ignored by the host.

Images (host side, per head; the K image repeats the S keys once per tile and pass, ``tiles x 2``):
    Q  [head][tile][4608 B]                       one A object a tile (``a_image``)
    K  [head][tile][pass][band][4608 B]           one B object a band (``b_image``, 64 keys x 64 dims)
    V  [head][tile][band][5760 B]                 one B object a band (V^T, 80 dims x 64 keys, dim 64 = 1.0)
    OUT[head][tile][8192 B]                       bf16 rows, 64 queries x 64 dims
"""

from pathlib import Path

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from artlite.gemm.mac import render_mac
from artlite.operators import kmath, templating
from artlite.operators.base import ArgSpec, ArtcOperator
from artlite.operators.design import MAX_BD_ITER
from artlite.operators.kernel_source import kernel_prefix
from artlite.vision import layout as L

from .moss_stem import a_image

QT = 32  # queries of a tile (one A object's 16-row bands; 32 keeps the core's L1: the fp32 score stripe and o)
KB = 64  # keys of a band (one B object)
KPAD = 64  # k-tiles' K (a head's real dims, the MAC's operand width)
HD, HP = 64, 80  # the head's dims, and their pad to 80 (the residual width 1280 = 16 heads x 80)
HEADS = 16
PER_CORE = 1  # heads a core takes per dispatch (the shim's 12 input channels and the memtile's 48 BDs: 4)
CORE_HEADS = 4  # heads a dispatch covers (one per core)
A_OBJ = QT * KB * 9 // 8  # QT x 64 keys of bfp16 panels: a Q object (a query tile)
K_OBJ = KB * HD * 9 // 8  # 4608 B: a K object (64 keys x 64 dims of bfp16 panels) at QT 64; the KV object's first half
V_OBJ = HP * KB * 9 // 8  # 5760 B: a V^T object (80 dims x 64 keys)
OUT_OBJ = QT * HD * 2  # 8192 B: bf16 rows of a tile's context
LOG2E = np.float32(1.4426950408889634)
_JINJA = Path(__file__).resolve().with_name("moss_attn.cc.jinja")
TEMPLATE = templating.JINJA_ENV.from_string(_JINJA.read_text())


# ------------------------------------------------------------------------------------------------------------- host side

def b_image(x: np.ndarray) -> np.ndarray:
    """(n, K) -> a B object's bfp16 bytes in the MAC's pop order [16-output block][k-step][half] (K = one k-tile)."""
    n, kk = x.shape
    assert n % 16 == 0 and kk == KPAD, (n, kk)
    p = L.panels(x).reshape(kk // 8, n // 8, L.PANEL)                 # [k-block][n panel][72 B]
    v = p.reshape(kk // 8, n // 16, 2, L.PANEL)                       # [k-step][block][half][72]
    return np.ascontiguousarray(v.transpose(1, 0, 2, 3)).reshape(-1)


def pad_keys(x: np.ndarray, s: int, q: int = KB) -> np.ndarray:
    """(S, d) rows -> (ceil(S / q) * q, d): the rows past S are zeros (their P is masked by the tail kernel)."""
    n = -(-s // q) * q
    out = np.zeros((n, x.shape[1]), np.float32)
    out[:s] = x
    return out


def pad_rows(x: np.ndarray, s: int, q: int = QT) -> np.ndarray:
    """(S, d) -> (ceil(S / q) * q, d) with zeros: padded queries are harmless (each query stands alone)."""
    return pad_keys(x, s, q)


def q_image(q: np.ndarray, tile0: int = 0, tiles: int = 0) -> np.ndarray:
    """(S, heads, 64) -> Q image bytes: per head and 64-query tile one A object (padded rows zero), tiles tile0 .."""
    s, heads, hd = q.shape
    assert hd == HD
    ntile = -(-s // QT)
    tiles = tiles or ntile
    out = []
    for h in range(heads):
        qp = pad_rows(q[:, h, :], s)
        for t in range(tile0, min(tile0 + tiles, ntile)):
            out.append(a_image(np.ascontiguousarray(qp[t * QT:(t + 1) * QT])))
    return np.concatenate(out)


def k_image(k: np.ndarray, tiles: int = 0) -> np.ndarray:
    """(S, heads, 64) -> K panels per head and band (padded keys' rows zero): [head][band][A_OBJ]."""
    s, heads, hd = k.shape
    assert hd == HD
    bands = -(-s // KB)
    out = []
    for h in range(heads):
        kp = pad_keys(np.ascontiguousarray(k[:, h, :]), s)
        out += [b_image(np.ascontiguousarray(kp[b * KB:(b + 1) * KB])) for b in range(bands)]
    return np.concatenate(out)


def v_image(v: np.ndarray, tiles: int = 0) -> np.ndarray:
    """(S, heads, 64) -> V^T panels per head and band, dim 64 a column of 1.0 (the row sums): [head][band][V_OBJ]."""
    s, heads, hd = v.shape
    assert hd == HD
    bands = -(-s // KB)
    out = []
    for h in range(heads):
        vp = pad_keys(np.ascontiguousarray(v[:, h, :]), s)
        for b in range(bands):
            band = np.ascontiguousarray(vp[b * KB:(b + 1) * KB])      # (64 keys, 64 dims)
            vt = np.zeros((HP, KB), np.float32)
            vt[:HD] = band.T                                          # (dims, keys)
            vt[HD] = 1.0
            out.append(b_image(vt))
    return np.concatenate(out)


def kv_image(k: np.ndarray, v: np.ndarray, tiles: int = 0) -> np.ndarray:
    """(S, heads, 64) k and v -> the KV image: one object a band holding K's A_OBJ panels then the V^T's V_OBJ
    panels, in the core's order [head][tile][pass][band] (the pass-1 objects' V halves are read and not used: the
    core has two input channels and Q takes the other)."""
    kk, vv = k_image(k), v_image(v)
    s, heads, _hd = k.shape
    bands = -(-s // KB)
    ntile = -(-s // QT)
    tiles = tiles or ntile
    out = []
    for h in range(heads):
        objs = [np.concatenate([kk[(h * bands + b) * K_OBJ:(h * bands + b + 1) * K_OBJ],
                                vv[(h * bands + b) * V_OBJ:(h * bands + b + 1) * V_OBJ]])
                for b in range(bands)]
        for _t in range(tiles):
            out += objs + objs
    return np.concatenate(out)


def from_out(y: np.ndarray, heads: int, s: int) -> np.ndarray:
    """OUT image bytes -> (S, heads, 64) fp32 (the padded queries dropped)."""
    tiles = -(-s // QT)
    v = np.asarray(y).view(bfloat16).reshape(heads, tiles, QT, HD).astype(np.float32)
    return np.ascontiguousarray(np.transpose(v, (1, 0, 2, 3))).reshape(heads * tiles * QT, HD)[:s * heads].reshape(
        heads, s, HD).transpose(1, 0, 2)


def quantize(x: np.ndarray) -> np.ndarray:
    """bfp16ebs8 round trip through the panel layout (the operands the MAC reads)."""
    m, c = x.shape
    return L.from_panels(L.panels(np.asarray(x, np.float32)), m, c)


def attn_walk(q: np.ndarray, k: np.ndarray, v: np.ndarray, *, s: int) -> np.ndarray:
    """The operator's math as a host model (exact per-query max, fp32 exp2, P rounded to bfp16, the row sums on V's pad
    column, fp32 accumulate, the finalize's division), for the gate."""
    heads, hd = q.shape[1], q.shape[2]
    out = np.zeros((s, heads, hd), np.float64)
    for h in range(heads):
        qh = pad_rows(np.ascontiguousarray(q[:, h, :]), s)
        kh = pad_keys(np.ascontiguousarray(k[:, h, :]), s)
        vh = pad_keys(np.ascontiguousarray(v[:, h, :]), s)
        bands = kh.shape[0] // KB
        ks = [quantize(np.ascontiguousarray(kh[b * KB:(b + 1) * KB])) for b in range(bands)]
        for t in range(qh.shape[0] // QT):
            qq = quantize(np.ascontiguousarray(qh[t * QT:(t + 1) * QT]))
            m = np.full(QT, -np.inf)
            for kb in ks:
                m = np.maximum(m, (qq @ kb.T).max(1))
            o = np.zeros((QT, HP))
            for b, kb in enumerate(ks):
                sc = qq @ kb.T
                p = np.exp2((sc - m[:, None]) * float(LOG2E))
                if b == bands - 1 and s % KB:
                    p[:, s % KB:] = 0.0                                            # (the tail band's mask)
                p = np.asarray(quantize(p), np.float64)                            # (bfp16 panels round trip)
                vb = np.ascontiguousarray(vh[b * KB:(b + 1) * KB])
                o[:, :HD] += p @ vb
                o[:, HD] += p.sum(1)
            inv = 1.0 / o[:, HD]
            nr = min(QT, s - t * QT)                                             # (the last tile's valid queries)
            out[t * QT:t * QT + nr, h, :] = (o[:nr, :HD] * inv[:nr, None])
    return out[:s].astype(np.float32)


# ------------------------------------------------------------------------------------------------------- the operator

class MossAttnOperator(ArtcOperator):
    """A layer's self-attention (module docstring). Arguments: Q (S x heads x 64 image bytes), K (the tile/pass/band
    image), V (V^T image) -> OUT (heads/tiles bf16 rows)."""

    p = kernel_prefix("mattn")
    STACK = 2048
    AIECC_FLAGS: tuple[str, ...] = ()

    def __init__(self, *, S: int, heads: int = CORE_HEADS, cores: int = CORE_HEADS, tiles: int = 12):
        assert heads == cores and heads <= HEADS and S > 0, (heads, cores, S)
        assert tiles > 0, (S, tiles)
        self.S, self.heads, self.cores = S, heads, cores
        self.tiles = min(tiles, -(-S // QT))  # (a dispatch covers at most all of its query tiles)
        self.nfull = S // KB          # (whole key bands)
        self.nv = S % KB              # (valid keys of the tail band, 0: none)
        self.bands = self.nfull + (1 if self.nv else 0)
        super().__init__()

    def exp2_fn(self) -> object:
        return kmath.exp2(32, prec="fp32")

    def get_kernel_src(self) -> str:
        p = self.p
        exp2 = self.exp2_fn()
        mac_z = render_mac(name=f"{p}_qk_t", k=KB, f32=True, zero=True)
        mac_p = render_mac(name=f"{p}_pv_t", k=KB, f32=True)
        nv = self.nv
        full = (1 << 32) - 1
        mask0 = 0 if nv >= 32 else full ^ ((1 << nv) - 1) if nv else full
        mask1 = 0 if nv >= 64 else full if nv <= 32 else full ^ ((1 << (nv - 32)) - 1)
        body = TEMPLATE.render(p=p, exp2=str(exp2), hd=HD, hp=HP, qt=QT, kb=KB, nv=nv, ovals=HP * QT,
                               k_obj=K_OBJ, log2e=float(LOG2E), mask0=mask0, mask1=mask1)
        return mac_z + "\n" + mac_p + "\n" + exp2.src + "\n" + body + f"""
extern "C" {{
// scores = Q K^T of a tile (fp32, C stripe), then context = P V^T accumulating into the tile's fp32 o
void {p}_qk(const bfloat16 *__restrict q, const bfloat16 *__restrict k, float *__restrict sc)
{{
    aie::set_rounding(aie::rounding_mode::conv_even);
    {p}_qk_t<{QT}, {KB}>((const bfp16ebs8 *)q, (const void *)k,
        (const float __aie_dm_resource_c *)sc, (float __aie_dm_resource_c *)sc);
}}
void {p}_pv(const bfloat16 *__restrict pp, const bfloat16 *__restrict vt, float *__restrict o)
{{
    aie::set_rounding(aie::rounding_mode::conv_even);
    {p}_pv_t<{QT}, {HP}>((const bfp16ebs8 *)pp, (const void *)vt,
        (const float __aie_dm_resource_c *)o, (float __aie_dm_resource_c *)o);
}}
}}
"""

    def get_arg_spec(self) -> list[ArgSpec]:
        tiles, heads, bands = self.tiles, self.heads, self.bands
        kv = K_OBJ + V_OBJ
        return [ArgSpec("in", (heads * tiles * A_OBJ // 2,), bfloat16),
                ArgSpec("in", (heads * tiles * 2 * bands * kv // 2,), bfloat16),
                ArgSpec("out", (heads * tiles * OUT_OBJ // 2,), bfloat16)]

    def get_mlir_artifact(self) -> Module:
        bf = np.dtype[bfloat16]
        tiles, bands, nfull, nv, heads = self.tiles, self.bands, self.nfull, self.nv, self.heads
        Q_ty = np.ndarray[(A_OBJ // 2,), bf]
        KV_OBJ = K_OBJ + V_OBJ
        KV_ty = np.ndarray[(KV_OBJ // 2,), bf]
        O_ty = np.ndarray[(OUT_OBJ // 2,), bf]
        SC_ty = np.ndarray[(QT * KB,), np.dtype[np.float32]]
        OV_ty = np.ndarray[(HP * QT,), np.dtype[np.float32]]
        RUN_ty = np.ndarray[(QT,), np.dtype[np.float32]]
        PP_ty = np.ndarray[(A_OBJ // 2,), bf]
        Q_img = np.ndarray[(heads * tiles * A_OBJ // 2,), bf]
        KV_img = np.ndarray[(heads * tiles * 2 * bands * KV_OBJ // 2,), bf]
        OUT_img = np.ndarray[(heads * tiles * OUT_OBJ // 2,), bf]
        p, cores = self.p, self.cores
        k_init = Kernel(f"{p}_init", "kernel.o", [OV_ty, RUN_ty])
        k_qk = Kernel(f"{p}_qk", "kernel.o", [Q_ty, KV_ty, SC_ty])
        k_max = Kernel(f"{p}_max", "kernel.o", [SC_ty, RUN_ty])
        k_exp = Kernel(f"{p}_exp", "kernel.o", [SC_ty, RUN_ty, PP_ty])
        k_expz = Kernel(f"{p}_expz", "kernel.o", [SC_ty, RUN_ty, PP_ty]) if nv else None
        k_pv = Kernel(f"{p}_pv", "kernel.o", [PP_ty, KV_ty, OV_ty])
        k_fin = Kernel(f"{p}_fin", "kernel.o", [OV_ty, O_ty])
        # a core has two input DMA channels: Q, and one KV object a band (K's panels then V^T's)
        q_l3 = [ObjectFifo(Q_ty, name=f"ma_q{c}", depth=2) for c in range(cores)]
        q_l1 = [q_l3[c].cons().forward(obj_type=Q_ty, name=f"ma_q1_{c}", depth=1, tile=Tile(c, 1)) for c in range(cores)]
        kv_l3 = [ObjectFifo(KV_ty, name=f"ma_kv{c}", depth=2) for c in range(cores)]
        kv_l1 = [kv_l3[c].cons().forward(obj_type=KV_ty, name=f"ma_kv1_{c}", depth=1, tile=Tile(c, 1))
                 for c in range(cores)]
        o_l3 = [ObjectFifo(O_ty, name=f"ma_o{c}", depth=2) for c in range(cores)]

        def body_(q_f, kv_f, o_f, init, qk, mx, ex, pv, fin, sc, ov, run, pp, exz):
            """One tile: the exact max pass, then the P/PV pass (``exz``: the tail band's masked P)."""
            for _h in range_(PER_CORE):
                for _t in range_(tiles):
                    q = q_f.acquire(1)
                    init(ov, run)
                    for _b in range_(nfull):
                        kv = kv_f.acquire(1)
                        qk(q, kv, sc)
                        mx(sc, run)
                        kv_f.release(1)
                    if nv:
                        kv = kv_f.acquire(1)
                        qk(q, kv, sc)
                        mx(sc, run)
                        kv_f.release(1)
                    for _b in range_(nfull):
                        kv = kv_f.acquire(1)
                        qk(q, kv, sc)
                        ex(sc, run, pp)
                        pv(pp, kv, ov)                       # (the same object: V^T at its second half)
                        kv_f.release(1)
                    if nv:
                        kv = kv_f.acquire(1)
                        qk(q, kv, sc)
                        exz(sc, run, pp)
                        pv(pp, kv, ov)
                        kv_f.release(1)
                    out = o_f.acquire(1)
                    fin(ov, out)
                    o_f.release(1)
                    q_f.release(1)

        def core(q_f, kv_f, o_f, init, qk, mx, ex, pv, fin, sc, ov, run, pp):
            for _ in forever_():
                body_(q_f, kv_f, o_f, init, qk, mx, ex, pv, fin, sc, ov, run, pp, None)

        def core_z(q_f, kv_f, o_f, init, qk, mx, ex, exz, pv, fin, sc, ov, run, pp):
            for _ in forever_():
                body_(q_f, kv_f, o_f, init, qk, mx, ex, pv, fin, sc, ov, run, pp, exz)

        # (an object's run is split into <= 1023-word inner runs: a shim / memtile BD's length)
        def obj_tap_n(n_obj: int, base: int, obj_elems: int, parts: int, total: int) -> TensorAccessPattern:
            inner = obj_elems // parts
            return TensorAccessPattern((total,), base, [n_obj, parts, inner], [obj_elems, inner, 1])

        def seq(Q, KV, OUT, qp, kp, op):
            tg = TaskGroup()
            for c in range(cores):
                qp[c].fill(Q, obj_tap_n(PER_CORE * tiles, c * PER_CORE * tiles * A_OBJ // 2, A_OBJ // 2, 3,
                                        heads * tiles * A_OBJ // 2), group=tg, wait=False)
                nkv = PER_CORE * tiles * 2 * bands
                for b0 in range(0, nkv, MAX_BD_ITER):
                    n = min(MAX_BD_ITER, nkv - b0)
                    kp[c].fill(KV, obj_tap_n(n, c * PER_CORE * tiles * 2 * bands * (KV_OBJ // 2) + b0 * KV_OBJ // 2,
                                             KV_OBJ // 2, 12, heads * tiles * 2 * bands * KV_OBJ // 2),
                               group=tg, wait=False)
            tg.finish()
            tgd = TaskGroup()
            for c in range(cores):
                op[c].drain(OUT, obj_tap_n(PER_CORE * tiles, c * PER_CORE * tiles * OUT_OBJ // 2, OUT_OBJ // 2, 8,
                                           heads * tiles * OUT_OBJ // 2), group=tgd, wait=True)
            tgd.finish()

        workers = []
        for c in range(cores):
            args = [q_l1[c].cons(), kv_l1[c].cons(), o_l3[c].prod(), k_init, k_qk, k_max, k_exp,
                    *([k_expz] if nv else []), k_pv, k_fin,
                    Buffer(SC_ty, name=f"ma_sc{c}"), Buffer(OV_ty, name=f"ma_ov{c}"), Buffer(RUN_ty, name=f"ma_run{c}"),
                    Buffer(PP_ty, name=f"ma_pp{c}")]
            workers.append(Worker(core_z if nv else core, args, tile=Tile(c, 2), stack_size=self.STACK,
                                  dynamic_objfifo_lowering=True, while_true=False))
        # (the shim's 2 S2MM per tile: Q and the KV stream on tile c; the context drains on tile c + 4)
        qp = [q_l3[c].prod(tile=Tile(c, 0)) for c in range(cores)]
        kp = [kv_l3[c].prod(tile=Tile(c, 0)) for c in range(cores)]
        op = [o_l3[c].cons(tile=Tile(c + 4, 0)) for c in range(cores)]
        return Program(NPU2(), Runtime(seq, [Q_img, KV_img, OUT_img, qp, kp, op]),
                       workers=workers).resolve_program()
