"""The Gated DeltaNet layers of a qwen35 prompt chunk of T tokens (the prefill), on the decode overlay's state
(``overlay.py``: GSTATE, the conv state in GHDR) and the gdn_in GEMMs' output G (``prefill.gemm``: column c's slice
[qkv | z | ab] at c pitch). Three operators per layer:

``GdnPrepOperator``: the q / k side, once per k head: per column two cores per k head (a k core and a q core, each
through its half of the head's input objects: the memtile splits them), per 8 tokens the conv of the head's k (or q)
channels (128), SiLU, l2 norm (q times 1 / sqrt(D): the scores' scale), as MMUL panels of the token block (bfp16 hi +
lo of each [token 8][i 8] block, ~14 bits); the q core also the gates of the k head's v heads, d = expm1(g) and beta,
times a host-written token mask (d = beta = 0 for padding tokens: the scan leaves the state alone through them). Per
16-token chunk the memtile joins the two cores' objects into one record ``JOBJ``: [K panels | Q panels | gates] in PR.

``GdnScanOperator``: the delta rule, chunkwise (the WY / UT form): per column four cores (rows 2 .. 5); a round per v
head of the column, core r taking the head's state quarter r (columns 32 r .. 32 r + 31: 16 KB of fp32, resident in L1
for the round). Per chunk of C = 16 tokens from state S0 (u = s + 1 .. t, s = -1: from the chunk's start):

- E[t][s] = prod_u (1 + d_u) - 1 (a scan of products: no cancellation for weak decays, exact zero for none);
- L = beta (1 + E) K K^T (s < t), M = (1 + E) Q K^T (s <= t), T = (I + L)^-1 per 8-token block (row substitution);
- X^T = (K S0)^T, Y^T = (Q S0)^T on the MMUL (S0 as bfp16 hi + lo against the K / Q panels: three products);
- R^T = beta (V^T - Gamma X^T); U^T = R^T T^T by blocks (U_1 = T11 (R_1 - L10 U_0)); O^T = Gamma Y^T + U^T M^T;
- S = S0 + (Gamma_C - 1) S0 + K^T (Gamma_C / Gamma U) (the accumulator starts at S0: exact fp32).

Every product on the MMUL takes both operands as bfp16 hi + lo (three products): o within ~3e-5 of the exact
recurrence (a token-by-token form: ~1.5e-5), 70x below its bf16 rounding. Per column: the conv (per 8 tokens) and SiLU
of the core's v quarter (32 channels), the conv state the 3 rows of G before the chunk (CV: G's rows before a sub-chunk,
or a layer's own rows). Streams per column:

- B (broadcast to the column's cores): per round the k head's records (PR), one per chunk;
- V (split, 1 KB parts): per round the v conv weights, the v conv state (CV, 16 rows read), the state quarters (16
  parts), per chunk the v quarters (G);
- O (joined, 1 KB parts): per chunk o (the head's 128 values, bf16, into the O buffer at the head's place of the
  ssm_out input), then the state quarters back (GSTATE).

``GdnOutOperator``: one core per column: per 8 tokens and pair of the column's v heads rms(o) ssm_norm silu(z) as the
ssm_out GEMM's bfp16 A panels; first the new conv state: the qkv rows of the last three valid tokens (GC: G's rows of
the whole prompt chunk, a runtime offset: ``conv_offset``) into GHDR.
"""

import math

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    ScratchpadParameter,
    TaskGroup,
    Worker,
    sync_parameters,
)
from aie.iron.controlflow import forever_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig
from artlite.operators import templating
from artlite.operators.base import ArgSpec, ArtcOperator
from artlite.operators.bfp16 import PANEL_BYTES, PANEL_ELEMS, panel_elems
from artlite.operators.chain import ChainPart, chained_fill
from artlite.operators.design import COLS, MAX_BD_ITER, TILE_BDS, check_bd_iterations
from artlite.operators.kernel_source import kernel_prefix
from artlite.operators.util import Attr, arg_elems, bf16_hilo, bf16_hilo_bits, mask64, regions
from artlite.prefill.gemm import Layout
from artlite.txn import HardwareLimitError

from .gdn_math import GDN, MATH, PANEL, D, GdnDims, helpers_source
from .image import gdn_v_heads

CONV_OFF = "gdn_conv"  # scratchpad parameter: GC's bf16 offset of the conv state rows (``conv_offset``)
C = 16  # tokens per chunk of the scan
B_ELEMS = 2560  # bf16 of a GdnPrep input object: 8 tokens x [ab slot 64 (alpha, beta) | q 128 | k 128]
QB_ELEMS = 1536  # (its q core's part: [ab | q]; the k core's: k)
PART = 512  # bf16 of a scan split / joined part (1 KB: a chunk's v or o quarter, 16 tokens x 32)
N_SPART = 4 * PANEL // (2 * PART)  # parts of a state quarter (16)
KP_BYTES = 16 * 2 * 2 * PANEL_BYTES  # a chunk's K (or Q) panels: [ib 16][tb 2][hi, lo]
PREP_P = 2 * B_ELEMS  # bytes of a (column, k head) record of P: the prep's first object [q part | k part]
V_P = 4 * 2 * PART  # bytes of a (column, v head) record of P: per quarter a part, its v conv weights (512 B) first
P_HEAD = 4096  # P: ssm_norm (fp32) first


def ab_f32(layout: Layout) -> bool:
    """gdn_in's alpha as fp32 (an F32 ``ssm_alpha``: its own block of plane pairs, ``prefill.gemm``), else bf16."""
    return "alpha" in layout.off


def ab_off(layout: Layout) -> int:
    """The column offset of the ab slot in gdn_in's output: alpha (2 HC bf16 units when fp32), then beta."""
    return layout.off["alpha"] if ab_f32(layout) else layout.off["ab"]


def jobj_bytes(vh: int) -> int:
    """Bytes of a chunk's record (GdnPrep's joined object): [K panels | Q panels | per v head of the k head d (16 fp32),
    beta (16 fp32)]."""
    return 2 * KP_BYTES + 128 * vh


def params_bytes(hc: int) -> int:
    """Bytes of P (``scan_params``) for HC v heads per column."""
    return P_HEAD + COLS * (2 * PREP_P + hc * V_P)


def scan_params(g: GGUFFile, cfg: QwenConfig, layer: int, rows: np.ndarray) -> np.ndarray:
    """P of a GDN layer (uint8): ssm_norm; per column and k head the prep's object [q part: q conv weights [tap 4][hi,
    lo][128] bf16, dt and ssm_a of the k head's v heads (fp32 [16] each) | k part: k conv weights]; per column and v
    head per quarter r its v conv weights [tap 4][hi, lo][32].
    ``rows``: the gdn_in qkv block's rows per column (``Block.rows``: global conv channels)."""
    hc = GdnDims(cfg).hc
    cw = np.asarray(g.tensor(f"blk.{layer}.ssm_conv1d.weight").raw, np.float32)  # (channels, 4)
    dt = np.asarray(g.tensor(f"blk.{layer}.ssm_dt.bias").raw, np.float32).reshape(-1)
    A = np.asarray(g.tensor(f"blk.{layer}.ssm_a").raw, np.float32).reshape(-1)
    norm = np.asarray(g.tensor(f"blk.{layer}.ssm_norm.weight").raw, np.float32).reshape(-1)
    out = np.zeros(params_bytes(hc), np.uint8)
    out[:norm.nbytes] = norm.view(np.uint8)

    def hl(w: np.ndarray) -> np.ndarray:  # (..., ch, 4) -> (..., [tap][hi, lo][ch]) bf16 bits as bytes
        hi, lo = bf16_hilo_bits(np.swapaxes(w, -1, -2))
        return np.stack([hi, lo], -2).view(np.uint8).reshape(*w.shape[:-2], -1)

    ch = np.stack([r[r >= 0] for r in rows])  # (COLS, channels): the columns' global conv channels
    prep = out[P_HEAD:P_HEAD + COLS * 2 * PREP_P].reshape(COLS, 2, PREP_P)
    for c in range(COLS):
        vh = gdn_v_heads(cfg, c)
        for kh in range(2):
            prep[c, kh, :2048] = hl(cw[ch[c, D * kh + np.arange(D)]])
            heads = vh[kh::2]
            prm = np.zeros(32, np.float32)
            prm[:len(heads)], prm[16:16 + len(heads)] = dt[heads], A[heads]
            prep[c, kh, 2048:2176] = prm.view(np.uint8)
            prep[c, kh, 3072:] = hl(cw[ch[c, 2 * D + D * kh + np.arange(D)]])
    h = np.arange(hc)
    v = ch[:, (4 * D + D * h[:, None] + np.arange(D)).reshape(hc, 4, 32)]  # (COLS, hc, quarter, 32)
    out[P_HEAD + COLS * 2 * PREP_P:].reshape(COLS, hc, 4, 2 * PART)[..., :512] = hl(cw[v])
    return out


_PREP = templating.load("models/qwen35/gdn_prefill.prep.cc.jinja")


class GdnPrepOperator(ArtcOperator):
    """The q / k side of a GDN layer's chunk, once per k head (module docstring): per column and k head a k core and
    a q core (rows 2 + 2 kh, 3 + 2 kh), each per 8 tokens the conv of its 128 channels, SiLU, the l2 norm and the
    token block's MMUL panels; the q core also the gates (d = expm1(g), beta) of the k head's v heads times the token
    mask. The memtile splits each input object between the two and joins their parts of a chunk's record (``JOBJ``).
    Runtime sequence: G (gdn_in's output, ``layout``), CV (``CV_ROWS`` rows of G's layout: the conv state in rows 0 ..
    2), P (``scan_params``), M (fp32 per token: 1 valid, 0 padding; ``MASK_ELEMS`` bf16 read), PR (the records:
    [column][k head][chunk]).

    Packed (``pack``: several inputs, segments, in the chunk; ``mask``): M holds per token 0 (padding) or 1 + its
    distance from its segment's first token (4 from 3 on), twice: from its start (the q core's part of the object) and
    from ``K_MASK`` (the k core's): a segment's first three tokens' convs again from its first token on (the zeros a
    sequence's first tokens see: their outputs as the segment's alone), d = -2 at its first token
    (``GdnScanOperator``'s reset there).

    Tree (``tree``: segments that each continue a prefix's state, the layer's conv state in CV; ``mask(.., tree=True)``):
    M as packed, segments starting at multiples of 16 tokens; a block that starts a segment takes the prefix's 3 rows as
    its window (saved from CV in a layer's first call, flagged after the mask values), and the record carries one more
    gate slot (``jobj_bytes(vh + 1)``) holding the mask values: ``GdnScanOperator``'s reload of the prefix's state."""

    STACK = 5120
    CV_ROWS = 12  # rows of CV read (the q part's 12 rows, the k part's 8: 3 used)
    MASK_ELEMS = B_ELEMS  # bf16 of the mask argument (an object; T fp32 used)
    K_MASK = QB_ELEMS // 2  # (packed: the fp32 of the mask's second copy, the k core's part of the object)
    pfx = kernel_prefix("gdp")

    def __init__(self, *, T: int, cfg: QwenConfig, layout: Layout, pack: bool = False, tree: bool = False):
        assert T % C == 0 and 4 * T <= 2 * QB_ELEMS and not (pack and tree)
        check_bd_iterations(T // 8, "GdnPrep")
        self.T, self.cfg, self.layout, self.pack, self.tree = T, cfg, layout, pack, tree
        self.hc = GdnDims(cfg).hc
        self.vh = self.hc // 2
        assert {"qkv", "z"} <= set(layout.off) and ("ab" in layout.off or {"alpha", "beta"} <= set(layout.off)) \
            and self.hc % 2 == 0 and self.hc <= 6, "two k heads of up to 3 v heads per column"
        super().__init__()

    def workspace(self) -> dict[str, int]:
        sizes = [("wc", 2 * 8 * 128), ("rows", 2 * 11 * 128), ("yc", 2 * 8 * 128), ("ys", 2 * 8 * 128),
                 ("xf", 4 * 8 * 128), ("gp", 4 * 32), ("mask", 4 * (self.T + 16))]
        return regions(sizes + ([("cv0", 2 * 3 * 128)] if self.tree else []))

    def get_kernel_src(self) -> str:
        return _PREP.render(p=self.pfx, g=GDN, gm=MATH, helpers=helpers_source(self.cfg.rms_norm_eps), w=Attr(self.workspace()),
                            D=D, HC=self.hc,
                            F32A=ab_f32(self.layout),
                            VH=self.vh, T=self.T, PAN=PANEL_BYTES, KP_BYTES=KP_BYTES, eps=repr(self.cfg.rms_norm_eps),
                            rs=bf16_hilo(1 / math.sqrt(D)), PACK=self.pack, TREE=self.tree,
                            SEG=self.pack or self.tree)

    @staticmethod
    def mask(n: int, T: int, lo: np.ndarray | None = None, t: int = 0, tree: bool = False) -> np.ndarray:
        """M (fp32 [T]) of a chunk of n valid tokens: 1 each, 0 the padding; packed (``lo``: per token its segment's
        first one, -1 for padding between segments), 1 + the distance from it up to 4, per call of ``t`` tokens its
        object ([MASK_ELEMS / 2] fp32: the values at 0 and at ``K_MASK``); ``tree``: segments from multiples of C
        tokens, the first call's flag after the values."""
        out = np.zeros(T, np.float32)
        if lo is None:
            out[:n] = 1
            return out
        seg = lo >= 0
        out[:n] = np.where(seg, 1 + np.minimum(np.arange(n) - lo, 3), 0)
        assert not tree or (lo[seg] % C == 0).all(), "tree: segments from multiples of C tokens"
        obj = np.zeros((T // t, GdnPrepOperator.MASK_ELEMS // 2), np.float32)
        obj[:, :t] = obj[:, GdnPrepOperator.K_MASK:GdnPrepOperator.K_MASK + t] = out.reshape(-1, t)
        if tree:  # (as int32 bits: any nonzero word)
            obj[0, t] = obj[0, GdnPrepOperator.K_MASK + t] = 1
        return obj.reshape(-1)

    @property
    def jobj(self) -> int:
        """bf16 of a chunk's record."""
        return jobj_bytes(self.vh + self.tree) // 2

    @property
    def n_pr(self) -> int:
        """bf16 of PR: per column and k head the chunk's records."""
        return COLS * 2 * self.T // C * self.jobj

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (self.T * self.layout.N,), bfloat16),
                ArgSpec("in", (self.CV_ROWS * self.layout.N,), bfloat16),
                ArgSpec("in", (params_bytes(self.hc) // 2,), bfloat16),
                ArgSpec("in", (self.MASK_ELEMS,), bfloat16),
                ArgSpec("out", (self.n_pr,), bfloat16)]

    def get_mlir_artifact(self) -> Module:
        p, T, lay = self.pfx, self.T, self.layout
        bf = np.dtype[bfloat16]
        W = self.workspace()
        kq = KP_BYTES // 2
        b_ty, qb_ty, kb_ty = (np.ndarray[(n,), bf] for n in (B_ELEMS, QB_ELEMS, B_ELEMS - QB_ELEMS))
        j_ty, ko_ty, qo_ty = (np.ndarray[(n,), bf] for n in (self.jobj, kq, self.jobj - kq))
        ws_ty = np.ndarray[(W["size"] // 4,), np.dtype[np.int32]]
        k_qinit = Kernel(f"{p}_qinit", "kernel.o", [qb_ty, ws_ty, np.int32])
        k_kinit = Kernel(f"{p}_kinit", "kernel.o", [kb_ty, ws_ty, np.int32])
        k_q = Kernel(f"{p}_qblock", "kernel.o", [qb_ty, ws_ty, qo_ty, np.int32, np.int32, np.int32])
        k_k = Kernel(f"{p}_kblock", "kernel.o", [kb_ty, ws_ty, ko_ty, np.int32, np.int32])
        i3 = [[ObjectFifo(b_ty, name=f"gp_i{c}_{kh}", depth=2) for kh in range(2)] for c in range(COLS)]
        i1 = [[i3[c][kh].cons().split([0, QB_ELEMS], tile=Tile(c, 1), depths=[2, 2], obj_types=[qb_ty, kb_ty],
                                      names=[f"gp_iq{c}_{kh}", f"gp_ik{c}_{kh}"]) for kh in range(2)]
              for c in range(COLS)]
        o3 = [[ObjectFifo(j_ty, name=f"gp_o{c}_{kh}", depth=2) for kh in range(2)] for c in range(COLS)]
        o1 = [[o3[c][kh].prod().join([0, kq], tile=Tile(c, 1), depths=[2, 2], obj_types=[ko_ty, qo_ty],
                                     names=[f"gp_ok{c}_{kh}", f"gp_oq{c}_{kh}"]) for kh in range(2)]
              for c in range(COLS)]
        n_ch = T // C

        def core_q(i_h, o_h, init, block, ws, kh):
            for _ in forever_():
                for k in range_(3):
                    b = i_h.acquire(1)
                    init(b, ws, k)
                    i_h.release(1)
                for i in range_(n_ch):
                    o = o_h.acquire(1)
                    for tb in range(2):
                        b = i_h.acquire(1)
                        block(b, ws, o, kh, i, tb)
                        i_h.release(1)
                    o_h.release(1)

        def core_k(i_h, o_h, init, block, ws):
            for _ in forever_():
                for k in range_(3):
                    b = i_h.acquire(1)
                    init(b, ws, k)
                    i_h.release(1)
                for i in range_(n_ch):
                    o = o_h.acquire(1)
                    for tb in range(2):
                        b = i_h.acquire(1)
                        block(b, ws, o, i, tb)
                        i_h.release(1)
                    o_h.release(1)

        workers = []
        for c in range(COLS):
            for kh in range(2):
                workers.append(Worker(core_k, [i1[c][kh][1].cons(), o1[c][kh][0].prod(), k_kinit, k_k,
                                               Buffer(ws_ty, name=f"gp_wk{c}_{kh}")],
                                      tile=Tile(c, 2 + 2 * kh), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
                workers.append(Worker(core_q, [i1[c][kh][0].cons(), o1[c][kh][1].prod(), k_qinit, k_q,
                                               Buffer(ws_ty, name=f"gp_wq{c}_{kh}"), kh],
                                      tile=Tile(c, 3 + 2 * kh), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
        N, pitch = lay.N, lay.pitch
        n_cv, n_p, n_pr = arg_elems(self, 1), arg_elems(self, 2), arg_elems(self, 4)
        per = n_ch * self.jobj  # (a column's k head's records)
        n_g = T // 8

        def seq(G, CV, P, M, PR, ih, oh):
            tg = TaskGroup()
            for c in range(COLS):
                qkv = c * pitch + lay.off["qkv"]
                for kh in range(2):
                    f = ih[c][kh]
                    f.fill(P, TensorAccessPattern((n_p,), (P_HEAD + (2 * c + kh) * PREP_P) // 2, [1, 1, 1, B_ELEMS],
                                                  [0, 0, 0, 1]), group=tg)
                    chained_fill(f, CV, [TensorAccessPattern((n_cv,), qkv + D * kh, [1, 1, 12, D], [0, 0, N, 1]),
                                         TensorAccessPattern((n_cv,), qkv + 2 * D + D * kh, [1, 1, 8, D], [0, 0, N, 1])],
                                 tg)
                    f.fill(M, TensorAccessPattern((self.MASK_ELEMS,), 0, [1, 1, 1, B_ELEMS], [0, 0, 0, 1]), group=tg)
                    chained_fill(f, G, [TensorAccessPattern((T * N,), c * pitch + ab_off(lay), [n_g, 1, 8, 64],
                                                            [8 * N, 0, N, 1]),
                                        TensorAccessPattern((T * N,), qkv + D * kh, [n_g, 1, 8, D], [8 * N, 0, N, 1]),
                                        TensorAccessPattern((T * N,), qkv + 2 * D + D * kh, [n_g, 1, 8, D],
                                                            [8 * N, 0, N, 1])], tg)
                    oh[c][kh].drain(PR, TensorAccessPattern((n_pr,), (2 * c + kh) * per, [1, n_ch, self.jobj // 64, 64],
                                                            [0, self.jobj, 64, 1]), group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(T * N,), bf], np.ndarray[(n_cv,), bf], np.ndarray[(n_p,), bf],
                           np.ndarray[(self.MASK_ELEMS,), bf], np.ndarray[(n_pr,), bf],
                           [[f.prod(tile=Tile(c, 0)) for f in fs] for c, fs in enumerate(i3)],
                           [[f.cons(tile=Tile(c, 0)) for f in fs] for c, fs in enumerate(o3)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


_SCAN = templating.load("models/qwen35/gdn_prefill.scan.cc.jinja")


class GdnScanOperator(ArtcOperator):
    """The delta rule of a GDN layer over a chunk of T tokens (module docstring), chunkwise on GdnPrep's records: the
    whole prompt chunk in one call (the states read and written once per chunk), its records those of T / ``sub``
    GdnPrep calls of ``sub`` tokens. Runtime sequence: PR (GdnPrep's records, [prep call][column][k head][chunk]), G
    (gdn_in's output, ``layout``: the v rows), CV (``CV_ROWS`` rows of G's layout: the conv state in rows 0 .. 2), P
    (``scan_params``: the v conv weights), GSTATE (the layer's state, per column its panels: read and written), O (T x
    hv x 128 bf16: o of every v head).

    Tree (``tree``: segments from multiples of C tokens, each continuing the prefix's state in GSTATE and conv state in
    CV, ``GdnPrepOperator(tree=True)``'s records): a core holds its prefix quarter (Sp, read once per round) beside the
    running one; a chunk that starts a segment (the record's mask slot) starts from Sp, its v window the prefix's rows;
    GSTATE is left as it was (no drain). The L1 for Sp: the cores' B, V and O FIFOs one object deep (the memtile's two),
    the chunk's state update after their release (``post``: the next record streams meanwhile)."""

    STACK = 3072
    CV_ROWS = 16  # rows of CV read (the v quarter's conv state: [16][32])
    RUN = MAX_BD_ITER * C  # tokens per BD of the v rows and o (a BD iterates at most MAX_BD_ITER times)
    MAX_RUNS = 4  # runs of the v rows and of o in a round (which queues 5 other shim BDs: B, a chain of 2, the state in and out)
    assert 5 + 2 * MAX_RUNS <= TILE_BDS
    pfx = kernel_prefix("gdsc")

    def __init__(self, *, T: int, cfg: QwenConfig, layout: Layout, sub: int = 0, ablate: tuple[str, ...] = (),
                 pack: bool = False, tree: bool = False):
        """``ablate`` (timing experiments, wrong results): "chunk" the chunk's math compiled out (the v conv and SiLU
        kept); "ev_chunk": an event window around a chunk call (results unchanged). ``pack``: several inputs
        (segments) in the chunk (GdnPrep's packed records: d = -2 at a segment's first token): the state reset there
        exactly (E's cut terms -1 exactly, Gamma X and Gamma Y exact zeros, the state zeroed where Gamma_C = 0), a
        segment's first three tokens' v convs from its first token on."""
        self.sub = sub or T
        self.pack, self.tree = pack, tree
        assert T % self.sub == 0 and not (pack and tree) and self.sub % C == 0
        check_bd_iterations(T // self.sub, "GdnScan: the sub-chunks")
        if T > self.MAX_RUNS * self.RUN:
            raise HardwareLimitError(f"GdnScan: T {T} needs more than {self.MAX_RUNS} runs of {self.RUN} tokens "
                                     f"(a round's shim BDs: at most {TILE_BDS})")
        self.T, self.cfg, self.layout = T, cfg, layout
        self.ablate = tuple(ablate)
        self.dims = GdnDims(cfg)
        self.hc, self.ch, self.hv = self.dims.hc, self.dims.ch, cfg.ssm_v_heads
        self.vh = self.hc // 2
        assert {"qkv", "z"} <= set(layout.off) and self.hc % 2 == 0 and self.hc <= 6
        super().__init__()

    def workspace(self) -> dict[str, int]:
        """(the gates' scratch g .. tr is dead once the T panels are written: xy and sw share its bytes; tree: the v
        conv's yc, ys too, dead once V^T is built)"""
        vconv = [("yc", 2 * 8 * 32), ("ys", 2 * 16 * 32)]
        common = [("wc", 2 * 4 * 2 * 32), ("rows", 2 * 11 * 32), *([] if self.tree else vconv),
                  ("vt", 2 * 4 * 2 * 64), ("kt", KP_BYTES), ("fv", 4 * 2 * 3 * 64), ("pn", 12 * PANEL_BYTES), ("dc", 2 * 2 * 64),
                  ("up", 4 * 4 * PANEL_BYTES)] + ([("pk", 4 * 32)] if self.pack else []) + \
            ([("cv0", 2 * 3 * 32), ("tf", 64)] if self.tree else [])
        gates = [("g", 4 * 6 * 64), ("er", 4 * 8 * 32 + 64), ("ed", 4 * 3 * 64), ("lt", 2 * 4 * 64), ("tr", 4 * 8 * 16),
                 ("rb", 2 * 2 * 8 * 16)]
        jblock = [("xy", 4 * 4 * 64), ("sw", 4 * 192 + 6 * PANEL_BYTES)]
        w = regions(common)
        shared = [regions(gates), regions(jblock), *([regions(vconv)] if self.tree else [])]
        out = {k: v for k, v in w.items() if k != "size"}
        for r in shared:
            out |= {k: w["size"] + v for k, v in r.items() if k != "size"}
        out["size"] = w["size"] + max(r["size"] for r in shared)
        return out

    def get_kernel_src(self) -> str:
        strict, lower = mask64(lambda t, s: s < t), mask64(lambda t, s: s <= t)
        return _SCAN.render(p=self.pfx, w=Attr(self.workspace()),
                            PART=PART, PAN=PANEL_BYTES, KP_BYTES=KP_BYTES, strict=f"{strict:016X}", lower=f"{lower:016X}",
                            ablate=self.ablate, PACK=self.pack, TREE=self.tree, VH=self.vh)

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (COLS * 2 * self.T // C * jobj_bytes(self.vh + self.tree) // 2,), bfloat16),
                ArgSpec("in", (self.T * self.layout.N,), bfloat16),
                ArgSpec("in", (self.CV_ROWS * self.layout.N,), bfloat16),
                ArgSpec("in", (params_bytes(self.hc) // 2,), bfloat16),
                ArgSpec("in", (COLS * self.dims.n_panels * PANEL // 2,), bfloat16),
                ArgSpec("out", (self.T * self.hv * D,), bfloat16)]

    def get_mlir_artifact(self) -> Module:
        p, T, hc, lay = self.pfx, self.T, self.hc, self.layout
        bf = np.dtype[bfloat16]
        W = self.workspace()
        jobj = jobj_bytes(self.vh + self.tree) // 2
        b_ty = np.ndarray[(jobj,), bf]
        part_ty = np.ndarray[(PART,), bf]
        quad_ty = np.ndarray[(4 * PART,), bf]
        ws_ty = np.ndarray[(W["size"] // 4,), np.dtype[np.int32]]
        S_ty = np.ndarray[(4 * PANEL // 4,), np.dtype[np.float32]]
        k_vinit = Kernel(f"{p}_vinit", "kernel.o", [part_ty, ws_ty, np.int32])
        k_sin = Kernel(f"{p}_sin", "kernel.o", [part_ty, S_ty, np.int32])
        k_sout = Kernel(f"{p}_sout", "kernel.o", [S_ty, part_ty, np.int32])
        k_chunk = Kernel(f"{p}_chunk", "kernel.o", [b_ty, part_ty, S_ty, ws_ty, part_ty, np.int32,
                                                    *([S_ty] if self.tree else [])])
        k_post = Kernel(f"{p}_post", "kernel.o", [S_ty, S_ty, ws_ty]) if self.tree else None

        b3 = [ObjectFifo(b_ty, name=f"gs_b{c}", depth=2) for c in range(COLS)]
        b1 = [b3[c].cons().forward(tile=Tile(c, 1), depth=2, name=f"gs_bm{c}") for c in range(COLS)]
        v3 = [ObjectFifo(quad_ty, name=f"gs_v{c}", depth=2) for c in range(COLS)]
        v1 = [v3[c].cons().split([PART * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4, obj_types=[part_ty] * 4,
                                 names=[f"gs_vp{c}_{r}" for r in range(4)]) for c in range(COLS)]
        o3 = [ObjectFifo(quad_ty, name=f"gs_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([PART * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[part_ty] * 4, names=[f"gs_op{c}_{r}" for r in range(4)]) for c in range(COLS)]
        n_ch = T // C

        def core(b_f, v_f, o_f, vinit, sin, sout, chunk, ws, S):
            for _ in forever_():
                for h in range_(hc):
                    for k in range_(2):
                        v = v_f.acquire(1)
                        vinit(v, ws, k)
                        v_f.release(1)
                    for k in range_(N_SPART):
                        v = v_f.acquire(1)
                        sin(v, S, k)
                        v_f.release(1)
                    for _ in range_(n_ch):
                        b = b_f.acquire(1)
                        v = v_f.acquire(1)
                        o = o_f.acquire(1)
                        chunk(b, v, S, ws, o, h)
                        o_f.release(1)
                        v_f.release(1)
                        b_f.release(1)
                    for k in range_(N_SPART):
                        o = o_f.acquire(1)
                        sout(S, o, k)
                        o_f.release(1)

        def core_tree(b_f, v_f, o_f, vinit, sin, chunk, post, ws, S, Sp):
            for _ in forever_():
                for h in range_(hc):
                    for k in range_(2):
                        v = v_f.acquire(1)
                        vinit(v, ws, k)
                        v_f.release(1)
                    for k in range_(N_SPART):
                        v = v_f.acquire(1)
                        sin(v, Sp, k)
                        v_f.release(1)
                    for _ in range_(n_ch):
                        b = b_f.acquire(1)
                        v = v_f.acquire(1)
                        o = o_f.acquire(1)
                        chunk(b, v, S, ws, o, h, Sp)
                        o_f.release(1)
                        v_f.release(1)
                        b_f.release(1)
                        post(S, Sp, ws)

        workers = []
        for c in range(COLS):
            for r in range(4):
                ws = Buffer(ws_ty, name=f"gs_ws{c}_{r}")  # (every byte written before it is read: no PDI zeroes)
                S = Buffer(S_ty, name=f"gs_S{c}_{r}")
                if self.tree:
                    args = [b1[c].cons(depth=1), v1[c][r].cons(depth=1), o1[c][r].prod(depth=1), k_vinit, k_sin,
                            k_chunk, k_post, ws, S, Buffer(S_ty, name=f"gs_Sp{c}_{r}")]
                else:
                    args = [b1[c].cons(), v1[c][r].cons(), o1[c][r].prod(), k_vinit, k_sin, k_sout, k_chunk, ws, S]
                workers.append(Worker(core_tree if self.tree else core, args,
                                      tile=Tile(c, 2 + r), stack_size=self.STACK,
                                      dynamic_objfifo_lowering=True,
                                      while_true=False))

        N, pitch = lay.N, lay.pitch
        n_pr, n_cv, n_p, n_st, n_o = (arg_elems(self, i) for i in (0, 2, 3, 4, 5))
        n_sub, sub_ch = T // self.sub, self.sub // C  # (GdnPrep calls, chunks per call)
        per, pr_sub = sub_ch * jobj, COLS * 2 * sub_ch * jobj
        runs = [(t0, min(self.RUN, T - t0)) for t0 in range(0, T, self.RUN)]

        def seq(PR, G, CV, P, GS, O, bh, vh, oh):
            for h in range(hc):  # (a round on every column at once)
                kh = h & 1
                tg = TaskGroup()
                for c in range(COLS):
                    qkv = c * pitch + lay.off["qkv"]  # (the column's conv channels: rows of CV and G alike)
                    vp = (P_HEAD + COLS * 2 * PREP_P + (c * hc + h) * V_P) // 2
                    st = (c * self.dims.n_panels + 16 * h) * PANEL // 2  # the head's state (4 quarters of 16 KB)
                    bh[c].fill(PR, TensorAccessPattern((n_pr,), (2 * c + kh) * per, [n_sub, sub_ch, jobj // 64, 64],
                                                       [pr_sub, jobj, 64, 1]), group=tg)
                    chained_fill(vh[c], P, [TensorAccessPattern((n_p,), vp, [1, 1, 4, PART], [0, 0, PART, 1]),
                                            ChainPart(CV, TensorAccessPattern((n_cv,), qkv + 4 * D + D * h, [1, 4, 16, 32],
                                                                              [0, 32, N, 1]))], tg)
                    vh[c].fill(GS, TensorAccessPattern((n_st,), st, [N_SPART, 4, 1, PART], [PART, 4 * PANEL // 2, 0, 1]),
                               group=tg)
                    j = gdn_v_heads(self.cfg, c)[h]
                    for t0, n in runs:
                        vh[c].fill(G, TensorAccessPattern((T * N,), t0 * N + qkv + 4 * D + D * h, [n // C, 4, 16, 32],
                                                          [16 * N, 32, N, 1]), group=tg)
                    for t0, n in runs:  # (tree: the round's last drain the one waited for; GS left as it was)
                        oh[c].drain(O, TensorAccessPattern((n_o,), t0 * self.hv * D + D * j, [n // C, 4, 16, 32],
                                                           [16 * self.hv * D, 32, self.hv * D, 1]), group=tg,
                                    wait=self.tree and t0 == runs[-1][0])
                    if not self.tree:
                        oh[c].drain(GS, TensorAccessPattern((n_st,), st, [N_SPART, 4, 1, PART],
                                                            [PART, 4 * PANEL // 2, 0, 1]), group=tg, wait=True)
                tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_pr,), bf], np.ndarray[(T * N,), bf], np.ndarray[(n_cv,), bf],
                           np.ndarray[(n_p,), bf], np.ndarray[(n_st,), bf], np.ndarray[(n_o,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(b3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(v3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()




_OUT = templating.load("models/qwen35/gdn_prefill.out.cc.jinja")


class GdnOutOperator(ArtcOperator):
    """The gated output of a GDN layer's chunk and its new conv state (module docstring). Per column 4 cores: the units
    (8 tokens x a pair of the column's v heads, pair-major) come as [o | z] objects, four per memtile object split
    between the cores, their panels joined back; the conv state goes from GC through the memtile into GHDR (no core).
    Runtime sequence: O (the scan's), G (z), GC (``gc_rows`` rows of G's layout: the prompt chunk's; the conv state rows
    from the scratchpad parameter ``gdn_conv``: ``conv_offset``; 3 rows: the conv state itself, no parameter), P
    (ssm_norm first), GHDR (the conv states written), A (the ssm_out GEMM's panels, T x hv 128)."""

    STACK = 4096
    OBJ = 2048  # bf16 of an o (or z) unit: 8 tokens x a pair of heads
    pfx = kernel_prefix("gdo")

    def __init__(self, *, T: int, cfg: QwenConfig, layout: Layout, hslot: int, cs_off: int, gc_rows: int = 0):
        assert T % 32 == 0
        check_bd_iterations(T // 8, "GdnOut")
        self.T, self.cfg, self.layout, self.hslot, self.cs_off = T, cfg, layout, hslot, cs_off
        self.gc_rows = gc_rows or T
        self.dims = GdnDims(cfg)
        self.hc, self.ch, self.hv = self.dims.hc, self.dims.ch, cfg.ssm_v_heads
        assert self.hc % 2 == 0 and 3 * self.ch <= 2 * self.OBJ
        assert cs_off + 4 * self.OBJ <= hslot, "the conv-state objects end in the column's slot"
        super().__init__()

    def get_kernel_src(self) -> str:
        p = self.pfx
        return _OUT.render(p=p, D=D, OBJ=self.OBJ, eps=repr(self.cfg.rms_norm_eps))

    def conv_offset(self, n: int) -> int:
        """The ``gdn_conv`` parameter of a chunk of n valid tokens: GC's first element of row n - 3."""
        assert 3 <= n <= self.gc_rows and self.gc_rows > 3, "the conv state: three tokens of the chunk"
        return (n - 3) * self.layout.N

    @property
    def ghdr_bytes(self) -> int:
        return COLS * self.hslot

    def get_arg_spec(self) -> list[ArgSpec]:
        T, hv = self.T, self.hv
        return [ArgSpec("in", (T * hv * D,), bfloat16),
                ArgSpec("in", (T * self.layout.N,), bfloat16),
                ArgSpec("in", (self.gc_rows * self.layout.N,), bfloat16),
                ArgSpec("in", (params_bytes(self.hc) // 2,), bfloat16),
                ArgSpec("out", (self.ghdr_bytes // 2,), bfloat16),
                ArgSpec("out", (panel_elems(T * hv * D),), bfloat16)]

    def get_mlir_artifact(self) -> Module:
        p, T, hc, lay, hv, ch = self.pfx, self.T, self.hc, self.layout, self.hv, self.ch
        bf = np.dtype[bfloat16]
        U = 2 * self.OBJ  # (a unit: [o | z])
        obj_ty, u_ty, uq_ty = (np.ndarray[(n,), bf] for n in (self.OBJ, U, 4 * U))
        pan_ty, panq_ty = np.ndarray[(32 * PANEL_ELEMS,), bf], np.ndarray[(4 * 32 * PANEL_ELEMS,), bf]  # 32 panels
        ws_ty = np.ndarray[(D + 8 * 256 + 4 * 256 + 512 + D // 32 * 32,), np.dtype[np.float32]]  # norm | y | silu(z) (bf16) | 1 / rms | norm hi / lo
        k_norm = Kernel(f"{p}_norm", "kernel.o", [u_ty, ws_ty])
        k_pair = Kernel(f"{p}_pair", "kernel.o", [u_ty, ws_ty, pan_ty])
        conv = ScratchpadParameter(CONV_OFF, np.int32) if self.gc_rows > 3 else None
        a3 = [ObjectFifo(uq_ty, name=f"go_a{c}", depth=2) for c in range(COLS)]
        a1 = [a3[c].cons().split([U * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4, obj_types=[u_ty] * 4,
                                 names=[f"go_ap{c}_{r}" for r in range(4)]) for c in range(COLS)]
        c3 = [ObjectFifo(panq_ty, name=f"go_c{c}", depth=2) for c in range(COLS)]
        c1 = [c3[c].prod().join([32 * PANEL_ELEMS * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[pan_ty] * 4, names=[f"go_cp{c}_{r}" for r in range(4)]) for c in range(COLS)]
        d3 = [ObjectFifo(obj_ty, name=f"go_d{c}", depth=2) for c in range(COLS)]  # (the conv state: DDR -> memtile -> DDR)
        d1 = [d3[c].cons().forward(tile=Tile(c, 1), depth=2, name=f"go_dm{c}") for c in range(COLS)]
        n_g, npair = T // 8, hc // 2
        per_core = n_g * npair // 4

        def core(a_h, c_h, norm, pair, ws):
            for _ in forever_():
                a = a_h.acquire(1)
                norm(a, ws)
                a_h.release(1)
                for _ in range_(per_core):
                    a = a_h.acquire(1)
                    c = c_h.acquire(1)
                    pair(a, ws, c)
                    c_h.release(1)
                    a_h.release(1)

        workers = [Worker(core, [a1[c][r].cons(), c1[c][r].prod(), k_norm, k_pair, Buffer(ws_ty, name=f"go_ws{c}_{r}")],
                          tile=Tile(c, 2 + r), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False)
                   for c in range(COLS) for r in range(4)]
        N = lay.N
        n_o, n_gc, n_p, n_hdr, n_a = (arg_elems(self, i) for i in (0, 2, 3, 4, 5))
        kb_row = hv * D // 8 * PANEL_ELEMS  # (bf16 of a token group's panels)


        def seq(O, G, GC, P, GH, A, ah, ch_, dh, d1h):
            if conv is not None:
                sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):
                # (the norm to each core: P's first unit read four times)
                ah[c].fill(P, TensorAccessPattern((n_p,), 0, [1, 4, U // 1024, 1024], [0, 0, 1024, 1]), group=tg)
                z0 = c * lay.pitch + lay.off["z"]
                for pp in range(npair):  # (the pair's units [o | z] of every token group)
                    chained_fill(ah[c], O, [TensorAccessPattern((n_o,), 2 * c * D + 16 * D * pp, [n_g, 1, 8, 2 * D],
                                                                [8 * hv * D, 0, hv * D, 1]),
                                            ChainPart(G, TensorAccessPattern((T * N,), z0 + 2 * D * pp, [n_g, 1, 8, 2 * D],
                                                                             [8 * N, 0, N, 1]))], tg)
                qkv = c * lay.pitch + lay.off["qkv"]
                if conv is not None:
                    dh[c].fill(GC, TensorAccessPattern((n_gc,), qkv, [1, 1, 3, ch], [0, 0, N, 1]), offset_parameter=conv,
                               group=tg)
                else:
                    dh[c].fill(GC, TensorAccessPattern((n_gc,), qkv, [1, 1, 3, ch], [0, 0, N, 1]), group=tg)
                dh[c].fill(GC, TensorAccessPattern((n_gc,), 0, [1, 1, 1, 2 * self.OBJ - 3 * ch], [0, 0, 0, 1]),
                           group=tg)
                # (the conv state and the objects' padding after it: the bytes the decode's conv-state drain writes)
                cs = (c * self.hslot + self.cs_off) // 2
                d1h[c].drain(GH, TensorAccessPattern((n_hdr,), cs, [1, 1, 1, 2 * self.OBJ], [0, 0, 0, 1]), group=tg)
                ch_[c].drain(A, TensorAccessPattern((n_a,), 2 * c * 16 * PANEL_ELEMS, [1, npair, n_g, 32 * PANEL_ELEMS],
                                                    [0, 16 * 16 * PANEL_ELEMS, kb_row, 1]), group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_o,), bf], np.ndarray[(T * N,), bf], np.ndarray[(n_gc,), bf],
                           np.ndarray[(n_p,), bf], np.ndarray[(n_hdr,), bf], np.ndarray[(n_a,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(a3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(c3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(d3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(d1)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()
