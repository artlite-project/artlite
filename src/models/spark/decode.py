"""Spark-X's decode design: the shared decode overlay of 8 slots (``decode.design``: the main and helper cores of every
column, their GEMV phases; GeGLU with GELU's erf form, the model's) and the head_dim-256 attention core at (c, 5) of every
column (``decode.attn256``, its design ``decode.attn256_design``) with Spark's spec (``attn_spec``)."""

from artlite.decode import attn256 as AT
from artlite.decode.attn256_design import Attn256DecodeOperator
from artlite.model_config import SparkConfig


def attn_spec(cfg: SparkConfig) -> AT.AttnSpec:
    """Spark's attention: no q / k norms, a sigmoid gate per head, the sliding-window layers' window."""
    return AT.AttnSpec(norms=False, gate="head", window=cfg.sliding_window, branch=False)


class SparkDecodeOperator(Attn256DecodeOperator):
    """The Spark-X decode overlay design: ``rows``: {tile format: rows per tile} of tiles of 1024 columns
    (``attn256.OBJ_BYTES``), ``attn``: the attention's shape (8 slots, ``attn_spec``); paged, gu_win, split, tails:
    ``decode.design``'s."""

    PREFIX = "spk"

    def __init__(self, *, hidden: int, rows: dict[str, int], attn: AT.AttnShape, eps: float,
                 paged: tuple[str, ...] = (), gu_win: bool = False, split: bool = False, tails: bool = False):
        assert attn.one and attn.spec.window, attn.spec
        super().__init__(attn=attn, hidden=hidden, rows=rows, eps=eps, nsb=4, act="gelu_erf", paged=paged, slots=8,
                         packed=True, bcast=True, gu_win=gu_win, split=split, tails=tails)
