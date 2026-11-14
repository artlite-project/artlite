"""The vision overlay's layers as ops of the sequencer (``seq``): per column its memtile tasks, per row its B stream.

By family: ``gemm`` (``Gemm``: y = x W^T + b (+ a residual) and its output kinds; ``conv``: its 3x3 conv variants and the
conv command's tap tables), ``apply`` (``Apply``: GroupNorm / LayerNorm into panels), ``attention`` (``Attention``,
``WinAttention``, ``DepthAttention``), ``dw`` (``DwConv``, ``StripConv``), ``common`` (the command builder, the MAC's
walks, the memtile bounds).

Tensors are memtile byte addresses, the same in every column; ``np`` is a panel tensor's token panels per channel block
(the column's tokens / 8, or more: a padded tensor), in (image row, group, image) order: token panel tp is image tp % 2.
"""

from .apply import Apply, Norm
from .attention import Attention, DepthAttention, WinAttention
from .common import ROPE2_TAB, ROPE40_TAB, ZERO_BYTES, Command, dims, gn_window, mac_walks
from .conv import ConvSpec, Wide
from .dw import DwConv, StripConv
from .gemm import Act, Gemm, Out, Stats

__all__ = [
           "ROPE2_TAB",
           "ROPE40_TAB",
           "ZERO_BYTES",
           "Act",
           "Apply",
           "Attention",
           "Command",
           "ConvSpec",
           "DepthAttention",
           "DwConv",
           "Gemm",
           "Norm",
           "Out",
           "Stats",
           "StripConv",
           "Wide",
           "WinAttention",
           "dims",
           "gn_window",
           "mac_walks",
]
