"""The DFlash2 block drafter's cores (``dflash``): a design of its own on the shared decode overlay (``decode.design``, 8 slots:
the main and helper cores as every model's), in a hardware context of its own, with

- the conv core (c, 2), which holds the block's residual: res [k block 80][slot 8][8] fp32 of the column's 640 channels. It
  emits x for the next normed phase, computes conv-in from res, the phase's rows (dyn) and the norm's sums of squares (``sq``,
  the helper's, in this core's memory), conv-out from the K-group sums of an output phase, into res. A group of 16 channels
  and its 2 + 2 taps' coefficients stay in one column: dyn's rows are the column's own (``dflash.conv_rows``).
- the attention core (c, 5): a KV head per column, its 4 query heads as octets of the 8 slots' rows (``attn``'s one-pass form: an
  MMUL panel of 8 rows, single bfp16 Q and P), non-causal over the block.

Per block position p + s a row attends to the ``window`` positions up to its own and to the whole block. A KV head's cache (the
buffer DKV) keeps closed records only, immutable, in a ring (``DFlashShape.ring``) of A objects, one per 32 positions (the prefill's
chunks start at multiples of it): [K | V] of 4 records, both in pairs [d-block][record 2] (``decode.kv``, kp = 1). The positions past the last closed record (at most 7 of the sequence, then the block's 8: always
inside two records) are one A object, the open object: exact rows [K | V][record & 1][d-block 16][position 8][8] bf16. The block's
K and V go into the core's copy of it and nowhere else; an injection (rows of the verified tokens) writes the object back, and a
record it closes into the ring, as K and V panels.

A step of the core (the F_ATT object, its rope object, the layer's norms, a q object per slot): the open object first, its pair of
records scored and added (the rows' references m are their maxima there: each row's own position is in it), then the ring's objects in
position order: scores, P, y += P V. The window's edge and the ring's slots past the closed records are
masked per lane (P exactly 0). At the end o = y / l goes to the main core (R), which relays it to C.

Conv core objects (``F_CV``, for the others an object of another core; ``M_CVOP``):

- ``CV_LOAD``: payload objects of res (fp32 [k block][slot 8][8]), the block's input rows; then x goes out (``M_CVNE`` D objects);
- ``CV_IN``: the object's payload [w 640 | b0 640 | b1 640] bf16 (the norm's weights, the conv's base of side 0, tap 0 and 1), then one
  object of dyn [slot 8][192] bf16 (rows [side 2][tap 2][40 groups]: side 1's stay for ``CV_OUT``): the phase's input x from res, x =
  res w / rms, out_s = (b0 + d0_s) x_s + (b1 + d1_s) x_(s - 1) as bf16 x on D;
- ``CV_OUT``: the payload [b0 640 | b1 640], then objects of y ([k block][slot 8][8] bf16, 32 k blocks each): res += (b0 + d0_s) y_s +
  (b1 + d1_s) y_(s - 1), then x from res on D.
"""

import numpy as np
from aie.iron import Buffer, Kernel, ObjectFifo, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.device import Tile
from ml_dtypes import bfloat16

from artlite.decode import design as OD
from artlite.decode.design import C_LEN, META, XPAY, DecodeOverlayOperator, _Streams
from artlite.decode.kv import Records
from artlite.model_config import DFlashConfig
from artlite.operators import templating
from artlite.operators.design import COLS
from artlite.operators.util import Attr, regions

from .attn import OBJ_BYTES

HD, QC, QH = 128, 4, 8 * 128  # (a head's dims, the query heads per column, the rows' stride of an octet)
ND = HD // 8
REC = Records(HD, 1)
REC_K, REC_V, ROWS_B = REC.k_bytes, REC.v_bytes, REC.rows_bytes
POS_PER_GROUP = 32  # an A object of the ring: 4 records [K pairs 2 | V pairs 2]
MAX_CHUNK = 2048  # a prefill chunk's positions at most
MAX_PADDING = 256  # a prefill chunk's padding rows at most (they write records too: over the ring's oldest slots, not the window's)
OPEN_BYTES = 4 * ROWS_B  # the open object's rows
PAIR = 2 * REC_K  # a pair of records
PO = 288  # an octet's P panels of 4 records
CHUNK = 4 * PO  # the 4 octets' P panels of 4 records
F_ATT = 1 << 10  # M_FLAGS of an attention step's first object
F_CV = 1 << 11  # M_FLAGS of a conv core's object
CV_LOAD, CV_IN, CV_OUT = range(3)
D_BYTES = 2048  # a conv core's D object: 16 k blocks of x
XK = 80  # k blocks of a column's x
H8 = 8 * XK  # a column's channels
GROUPS = H8 // 16  # conv groups of a column
DYN_ROWS = 192  # dyn's rows per slot in its object: whole C objects of its 160
M_CVOP, M_CVNE, M_POS, M_INJ, M_KEEP, M_ANOBJ, M_G0 = range(OD.M_MODEL, OD.M_MODEL + 7)
# conv core state words (the first ones are trip counts of its program)
V_NSKIP, V_NOBJ, V_NE, V_OP, V_I, V_IE = range(6)
# attention core state words (the first ones are trip counts of its program)
(A_NNO, A_NXN, A_NXSKIP, A_NASKIP, A_NSTEP, A_NQ, A_NE1, A_NOBJ, A_NE, A_XL, A_LPER, A_P, A_S, A_INJ, A_KEEP, A_I, A_G0, A_QS,
 A_EIDX, A_SOFT, A_NRM, A_ROPE, A_KO, A_VO) = range(24)
STATE_WORDS = 32
assert A_VO < STATE_WORDS
STACK = 1600
V_STACK = 1024
NE_BLOCK = QC * ND  # R objects of a block's o
NE_INJ = (OPEN_BYTES + 2 * REC_K) // (2 * C_LEN)  # R objects of an injection: the open object, a record's K and V panels

_CONV = templating.load("models/qwen35/dflash_core.conv.cc.jinja")

_ATT = templating.load("models/qwen35/dflash_core.att.cc.jinja")


class DFlashShape:
    """The cores' sizes for a drafter ``cfg`` (hidden 5120, 8 KV heads of 128, 4 query heads per KV head), the conv and attention
    workspaces, the ring of groups a window spans."""

    def __init__(self, cfg: DFlashConfig, *, a_bytes: int = OBJ_BYTES):
        assert cfg.head_dim == HD and cfg.kv_heads == COLS and cfg.heads == QC * COLS and cfg.hidden == COLS * H8
        assert cfg.block == 8 and cfg.conv_kernel == 2 and cfg.conv_group == 16, cfg
        assert 4 * PAIR <= a_bytes and OPEN_BYTES <= a_bytes
        assert 3 * H8 * 2 <= XPAY and 8 * DYN_ROWS * 2 <= XPAY and 8 * 128 * 4 <= XPAY
        self.cfg, self.a_bytes = cfg, a_bytes
        self.window = cfg.window
        # (objects: a window's closed records in distinct ones, what a chunk's padding rows overwrite past them)
        self.ring = -(-(cfg.window + 8 + MAX_PADDING) // POS_PER_GROUP) + 1
        self.over = MAX_CHUNK // POS_PER_GROUP  # (objects past the ring that a prefill chunk's writes reach)
        self.sets = 2 * QC
        self.eps = cfg.rms_norm_eps

    def conv_workspace(self) -> dict[str, int]:
        """The conv core's workspace: state words, res, the constants of an object, dyn, the slots' 1 / rms."""
        return regions([("st", 4 * 16), ("res", 4 * 8 * 8 * XK), ("cs", 2 * 3 * H8), ("dyn", 2 * 8 * DYN_ROWS), ("ir", 4 * 64)])

    def workspace(self) -> dict[str, int]:
        """The attention core's workspace. y: per set its blocks [db][row 4 x 8] fp32, then l (before the panels: the q rows
        [row][HD] fp32); sc: per octet the scores of 4 records [set 2][record 4][32] (before the cache: a row's K); pp: their P
        panels, [octet][P 4]; qa0, qa1: the Q panels [d-block][octet 2]."""
        return regions([("st", 4 * STATE_WORDS), ("y", 4 * 32 * (ND + 1) * self.sets), ("ml", 4 * 64 * self.sets),
                        ("sc", 4 * 128 * self.sets), ("pp", CHUNK), ("qa0", 2 * 72 * ND), ("qa1", 2 * 72 * ND)])

    def ext(self) -> dict[str, int]:
        """Regions in the main core's idle memory, past its residual: offsets from there."""
        return regions([("nrm", 4 * 2 * HD), ("rope", 4 * 128 * 8), ("ko", PAIR), ("vo", PAIR)])

    def render(self, p: str, consts: dict[str, int], *, tail: int) -> str:
        e = self.ext()
        at = [(A_NRM, tail + e["nrm"]), (A_ROPE, tail + e["rope"]), (A_KO, tail + e["ko"]), (A_VO, tail + e["vo"])]
        v = dict(p=p, a=Attr(self.workspace()), v=Attr(self.conv_workspace()), at=at, QC=QC, SETS=self.sets, HD=HD, ND=ND, PO=PO, CHUNK=CHUNK,
                 PAIR=PAIR, REC_K=REC_K, REC_V=REC_V, ROWS_B=ROWS_B, OPEN_BYTES=OPEN_BYTES, C_LEN=C_LEN, eps=repr(self.eps),
                 WINDOW=self.window, META=META, XPAY=XPAY, H8=H8, XK=XK, GROUPS=GROUPS, DYN_ROWS=DYN_ROWS, H=self.cfg.hidden,
                 NO_OBJ=NE_BLOCK, NE_INJ=NE_INJ, CV_LOAD=CV_LOAD, CV_IN=CV_IN, CV_OUT=CV_OUT,
                 **(consts | {"F_ATT": F_ATT, "F_CV": F_CV} | {k: w for k, w in globals().items() if k.startswith(("A_", "V_", "M_"))}))
        return _CONV.render(**v) + _ATT.render(**v)


class DFlashOperator(DecodeOverlayOperator):
    """The drafter's design (module docstring): ``rows``: {tile format: rows per tile} of tiles of ``tile_bytes``; ``shape``: its
    cores' sizes."""

    PREFIX = "dfl"
    A_FWD = 2
    LENDS_WS = True
    CONV_ROW, ATTN_ROW = 2, 5
    CONV_AT = 0x4000  # (the conv workspace: regions 1 .. 3, below the helper's scratch)

    def __init__(self, *, hidden: int, rows: dict[str, int], shape: DFlashShape, tile_bytes: int = OBJ_BYTES, eps: float = 1e-6):
        self.shape = shape
        self.relay = True
        super().__init__(hidden=hidden, rows=rows, tile_bytes=tile_bytes, eps=eps, nsb=4, slots=8, packed=True, bcast=True)

    def _consts(self) -> dict[str, int]:
        return super()._consts() | {k: v for k, v in globals().items() if k[:2] == "M_"}

    def _extra_src(self, p: str, consts: dict[str, int]) -> str:
        W = self.workspace()
        assert W["size"] - W["inv"] >= self.shape.ext()["size"], W
        return self.shape.render(p, {k: v for k, v in consts.items() if k[:2] in ("M_", "F_")}
                                 | {"NX_PER_NORM": self.nx_per_norm}, tail=W["inv"])

    @property
    def d_cols(self) -> list[int]:
        return list(range(COLS))

    def _pins(self) -> dict[str, int]:
        return {f"{self.PREFIX}_vws{c}": self.CONV_AT for c in range(COLS)}

    def _x_rows(self) -> tuple[int, ...]:
        return (self.CONV_ROW, self.ATTN_ROW)

    def _extra_workers(self, s: _Streams) -> tuple[list[Worker], list[tuple[int, ObjectFifo]]]:
        p, P, sh = s.p, self.PREFIX, self.shape
        bf, i32 = np.dtype[bfloat16], np.dtype[np.int32]
        x_ty, a_ty = s.x_ty, s.a_ty
        VW, AW = sh.conv_workspace(), sh.workspace()
        vws_ty = np.ndarray[(VW["size"] // 4,), i32]
        aws_ty = np.ndarray[(AW["size"] // 4,), i32]
        mws_ty = np.ndarray[(self.workspace()["size"] // 4,), i32]
        d_ty = np.ndarray[(D_BYTES // 2,), bf]
        sq_ty = np.ndarray[(8 * self.slots,), np.dtype[np.float32]]
        i_v, i_a = VW["st"] // 4, AW["st"] // 4
        k_vbegin = Kernel(f"{p}_vbegin", "kernel.o", [x_ty, vws_ty])
        k_vobj = Kernel(f"{p}_vobj", "kernel.o", [x_ty, vws_ty])
        k_vout = Kernel(f"{p}_vout", "kernel.o", [vws_ty, d_ty, sq_ty])
        k_abegin = Kernel(f"{p}_abegin", "kernel.o", [x_ty, aws_ty, mws_ty])
        k_anb = Kernel(f"{p}_anb", "kernel.o", [aws_ty])
        k_arope = Kernel(f"{p}_arope", "kernel.o", [x_ty, aws_ty])
        k_anorm = Kernel(f"{p}_anorm", "kernel.o", [x_ty, aws_ty])
        k_aq = Kernel(f"{p}_aq", "kernel.o", [x_ty, aws_ty, a_ty])
        k_aopen = Kernel(f"{p}_aopen", "kernel.o", [aws_ty, a_ty])
        k_aobj = Kernel(f"{p}_aobj", "kernel.o", [a_ty, aws_ty])
        k_aend = Kernel(f"{p}_aend", "kernel.o", [aws_ty])
        k_aemit = Kernel(f"{p}_aemit", "kernel.o", [aws_ty, s.r_ty])
        k_aemit1 = Kernel(f"{p}_aemit1", "kernel.o", [aws_ty, s.r_ty, a_ty])

        def conv_fn(x_f, d_f, vbegin, vobj, vout, ws, sq):
            for _ in forever_():
                x = x_f.acquire(1)
                vbegin(x, ws)
                x_f.release(1)
                for _ in range_(ws[i_v + V_NSKIP]):
                    x_f.acquire(1)
                    x_f.release(1)
                for _ in range_(ws[i_v + V_NOBJ]):
                    x = x_f.acquire(1)
                    vobj(x, ws)
                    x_f.release(1)
                for _ in range_(ws[i_v + V_NE]):
                    d = d_f.acquire(1)
                    vout(ws, d, sq)
                    d_f.release(1)

        def attn_fn(a_f, x_f, r_f, abegin, anb, arope, anorm, aq, aopen, aobj, aend, aemit, aemit1, ws, mws):
            for _ in forever_():
                x = x_f.acquire(1)
                abegin(x, ws, mws)
                x_f.release(1)
                for _ in range_(ws[i_a + A_NNO]):  # (a main core's normed phase: as it consumes them)
                    a_f.acquire(1)
                    anb(ws)
                    for _ in range_(ws[i_a + A_NXN]):
                        x_f.acquire(1)
                        x_f.release(1)
                    a_f.release(1)
                for _ in range_(ws[i_a + A_NXSKIP]):
                    x_f.acquire(1)
                    x_f.release(1)
                for _ in range_(ws[i_a + A_NASKIP]):
                    a_f.acquire(1)
                    a_f.release(1)
                for _ in range_(ws[i_a + A_NSTEP]):
                    x = x_f.acquire(1)
                    arope(x, ws)
                    x_f.release(1)
                    x = x_f.acquire(1)
                    anorm(x, ws)
                    x_f.release(1)
                    a = a_f.acquire(1)  # the open object
                    for _ in range_(ws[i_a + A_NQ]):
                        x = x_f.acquire(1)
                        aq(x, ws, a)
                        x_f.release(1)
                    aopen(ws, a)
                    for _ in range_(ws[i_a + A_NE1]):
                        r = r_f.acquire(1)
                        aemit1(ws, r, a)
                        r_f.release(1)
                    a_f.release(1)
                    for _ in range_(ws[i_a + A_NOBJ]):
                        a = a_f.acquire(1)
                        aobj(a, ws)
                        a_f.release(1)
                    aend(ws)
                    for _ in range_(ws[i_a + A_NE]):
                        r = r_f.acquire(1)
                        aemit(ws, r)
                        r_f.release(1)

        workers, d_fifos = [], []
        for c in range(COLS):
            d_fifos.append((c, ObjectFifo(d_ty, name=f"{P}_D{c}", depth=1)))
            vws = Buffer(vws_ty, name=f"{P}_vws{c}", initial_value=np.zeros(VW["size"] // 4, np.int32), tile=Tile(c, self.CONV_ROW))
            workers.append(Worker(conv_fn, [s.x_l1[c].cons(depth=1, **s.x_kw), d_fifos[c][1].prod(), k_vbegin, k_vobj, k_vout, vws, s.sq[c]],
                                  tile=Tile(c, self.CONV_ROW), stack_size=V_STACK, dynamic_objfifo_lowering=True, while_true=False))
            aws = Buffer(aws_ty, name=f"{P}_aws{c}", initial_value=np.zeros(AW["size"] // 4, np.int32), tile=Tile(c, self.ATTN_ROW))
            workers.append(Worker(attn_fn, [s.a_l1[c].cons(depth=2), s.x_l1[c].cons(depth=1, **s.x_kw), s.r_fifos[c].prod(), k_abegin, k_anb,
                                            k_arope, k_anorm, k_aq, k_aopen, k_aobj, k_aend, k_aemit, k_aemit1, aws, s.ws[c]],
                                  tile=Tile(c, self.ATTN_ROW), stack_size=STACK, dynamic_objfifo_lowering=True, while_true=False))
        return workers, d_fifos
