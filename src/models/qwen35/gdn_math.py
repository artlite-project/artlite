"""What the Gated DeltaNet's decode core (``gdn.py``) and prefill operators (``gdn_prefill.py``) share: a layer's
dimensions per column (``GdnDims``), the fp32 math (``MATH``: kmath, 16 lanes, one out-of-line exp2 for all of it; ``gm`` in
the templates) and the kernel helpers (``gdn_math.cc.jinja``, rendered with p, eps, D, gm and sigmoid, whether to define
it, in an anonymous namespace: the decode core includes them, a prefill operator renders ``helpers_source``; the helper
names are under ``GDN``, one copy for the operators of a fused program)."""

from artlite.model_config import QwenConfig
from artlite.operators import kmath, templating
from artlite.operators.design import COLS
from artlite.operators.kernel_source import kernel_prefix

D = 128  # ssm_state: k and v dims of a head
PANEL = 4096  # bytes of a state panel (128 i x 8 j fp32)

GDN = kernel_prefix("gdn")
EXP2 = kmath.exp2(16, prec="fp32")
MATH = {"exp2": EXP2, "expm1": kmath.expm1(16, exp_fn=EXP2), "softplus": kmath.softplus(16, exp_fn=EXP2),
        "sigmoid": kmath.sigmoid(16, exp_fn=EXP2), "silu": kmath.silu(32)}


def helpers_source(eps: float) -> str:
    """The helpers' source (with the sigmoid) for RMS norm epsilon ``eps``."""
    return templating.load("models/qwen35/gdn_math.cc.jinja").render(p=GDN, eps=repr(eps), D=D, gm=MATH, sigmoid=True)


class GdnDims:
    """A GDN layer's sizes per column for a model: HC v heads (pairs of them: even), CH conv channels, NB conv blocks of
    D channels, the column's state panels."""

    def __init__(self, cfg: QwenConfig):
        assert cfg.ssm_state == D and cfg.ssm_k_heads == 2 * COLS and cfg.ssm_v_heads % (2 * COLS) == 0
        self.hc = cfg.ssm_v_heads // COLS
        assert self.hc <= 8 and self.hc % 2 == 0
        self.ch = (4 + self.hc) * D
        self.nb = self.ch // D
        self.eps = cfg.rms_norm_eps

    @property
    def n_panels(self) -> int:
        """State panels of a column."""
        return 16 * self.hc
