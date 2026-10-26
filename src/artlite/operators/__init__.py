"""The operator framework: what every design is built with, whatever it computes.

An operator (``base.ArtcOperator``) renders its kernel source and its array design, compiles them (``kernel_source``:
the translation unit; ``checks``: the build-time checks; ``render_cache``: compiles without rendering what is built)
and dispatches them; ``fused`` puts several operators into one ELF. The pieces the operators share: ``design`` (aie.iron
helpers), ``l1_layout`` (pinned L1 buffers), ``kmath`` (math functions per contract), ``paging`` (code pages),
``bfp16`` (host conversions), ``util``; the kernels' headers are ``artlite/include``.
"""

from .base import ArgSpec
from .fused import FusedOperator

__all__ = [
    "ArgSpec",
    "FusedOperator",
]
