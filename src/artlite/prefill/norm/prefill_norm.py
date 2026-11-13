"""``PrefillNormOperator``: the (add +) RMSNorm of a prompt with bfp16ebs8 panel output (module docstring of
``prefill.norm``), in one of three designs by its options."""

import enum

from aie.ir import Module
from ml_dtypes import bfloat16

from ...operators import templating
from ...operators.base import ArgSpec, ArtcOperator
from ...operators.design import COLS
from ...operators.kernel_source import kernel_prefix
from .hold import HCH, HG, hold_design
from .plain import CH, plain_design
from .replay import replay_design

_SRC = templating.load("artlite/prefill/norm/prefill_norm.cc.jinja")
_HOLD_SRC = templating.load("artlite/prefill/norm/hold.cc.jinja")


class Design(enum.Enum):
    PLAIN = "plain"  # a group of 8 tokens in L1 next to the streams
    REPLAY = "replay"  # E > 2048, or two deltas: the memtile holds the group and sends it twice
    HOLD = "hold"  # more than two deltas: a group's sum held in core L1


DESIGNS = {Design.PLAIN: plain_design, Design.REPLAY: replay_design, Design.HOLD: hold_design}


class PrefillNormOperator(ArtcOperator):
    """(Add +) RMSNorm of a prompt with bfp16ebs8 panel output; see the module docstring."""

    STACK = 1024
    p = kernel_prefix("pn")

    def __init__(self, *, T: int, add: bool = True, E: int = 2048, eps: float = 1e-5, d_pitch: int = 0,
                 cols: int = COLS, deltas: int = 1, d_stride: int = 0, lo: bool = False, pre: bool = False,
                 out_pitch: int = 0, out_col: int = 0, raw: bool = False):
        """cols: cores (one per column) taking the token groups in turn: few for short prompts, where the
        op's reconfiguration costs more than its work. deltas = 2 (replay mode): x += d + d2 (a projection in two
        K parts: summed in fp32, rounded once), both at d_pitch, d2 ``d_stride`` elements after d
        in the delta argument (default T 8 d_pitch; e.g. the C size of a GEMM that pads the tokens). lo: xn as
        hi + lo panels (an MoE router's x), panels of 2 E columns: per token group [bfp16(v) | bfp16(v - bfp16(v))]
        (one delta, E <= 2048). pre (no add): the weight argument [w1 | w2], xn = (x w1) w2 / rms(x w1) (a norm of a
        norm's output: an MTP drafter's hnorm of the model's output norm). out_pitch, out_col (the plain design): xn's
        panel rows ``out_pitch`` columns wide (default E), these at columns out_col .. (another norm's beside them; not
        in the hold design). raw (no add): xn = x w, without the norm (a drafter's taps as panels: w of ones)."""
        assert T % (8 * cols) == 0 and T > 0 and E % CH == 0 and 1 <= cols <= COLS
        self.T, self.add, self.E, self.eps, self.cols, self.lo = T, add, E, eps, cols, lo
        self.d_pitch = d_pitch or E // COLS  # (pitch of delta's column blocks)
        assert self.d_pitch >= E // COLS and (add or not d_pitch)
        self.deltas = deltas if add else 0
        # more deltas: the hold design (the replay design's memtile holds a group's deltas twice over); the group
        # (8 tokens x E bf16) does not fit L1 next to the streams, or two deltas: the replay design's
        replay = self.deltas <= 2 and (16 * E > 32768 or (add and deltas == 2))
        self.design = Design.HOLD if self.deltas > 2 else Design.REPLAY if replay else Design.PLAIN
        assert self.deltas in (0, 1) or (self.deltas == 2 and self.replay) or self.hold
        assert not self.hold or (T % (8 * cols) == 0 and HG * E * 2 <= 40960
                                 and self.deltas * HG * E * 2 + 2 * E // 8 * 72 <= 500 * 1024), (T, E, deltas, cols)
        self.d_stride = d_stride or T * COLS * self.d_pitch
        assert self.d_stride >= T * COLS * self.d_pitch and (self.deltas >= 2 or not d_stride)
        self.ch = min(CH, E // 8)  # (columns per chunk)
        assert self.replay or self.hold or 8 * self.ch == E, "the norm weight: one chunk object"
        assert E % 1024 == 0 or self.replay, "the plain design: E of whole 1024 columns"
        assert not lo or not (self.replay or self.hold), "lo panels: the plain design"
        self.pre, self.raw = pre, raw
        assert not raw or not (add or pre or lo), "raw: x's own panels"
        self.out_pitch, self.out_col = out_pitch or E, out_col
        assert not pre or not add, "pre: a norm without a delta"
        assert (self.out_pitch, out_col) == (E, 0) or not (lo or self.hold) and (
            out_col % 8 == 0 and out_col + E <= self.out_pitch), (out_pitch, out_col)
        super().__init__()

    @property
    def hold(self) -> bool:
        return self.design is Design.HOLD

    @property
    def replay(self) -> bool:
        return self.design is Design.REPLAY

    def get_arg_spec(self) -> list[ArgSpec]:
        EMB = self.E
        n = self.T * EMB
        specs = [ArgSpec("in", (n,), bfloat16)]
        if self.add:  # (two deltas: one argument, [delta | delta2])
            specs.append(ArgSpec("in", ((self.deltas - 1) * self.d_stride + self.T * COLS * self.d_pitch,),
                                 bfloat16))
        specs.append(ArgSpec("in", ((2 if self.pre else 1) * EMB,), bfloat16))
        if self.add:
            specs.append(ArgSpec("out", (n,), bfloat16))
        specs.append(ArgSpec("out", ((2 if self.lo else 1) * self.T * self.out_pitch * 9 // 16,), bfloat16))
        return specs

    def get_kernel_src(self) -> str:
        if self.hold:
            return _HOLD_SRC.render(p=self.p, E=self.E, C=HCH, G=HG, eps=repr(self.eps), nd=self.deltas)
        # (the same source for add and no add: one copy in a fused ELF)
        return _SRC.render(p=self.p, E=self.E, C=self.ch, eps=repr(self.eps), replay=self.replay, lo=self.lo, pre=self.pre,
                           raw=self.raw)

    def get_mlir_artifact(self) -> Module:
        return DESIGNS[self.design](self)
