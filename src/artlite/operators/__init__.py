"""The operator framework: what every design is built with, whatever it computes.

An operator (``base.ArtcOperator``) renders its kernel source and its array design; ``build`` compiles them
(``toolchain``: the tools; ``kernel_source``: the translation unit; ``checks``: the build-time checks, over
``objdump``'s parse of the kernel object; ``render_cache``: compiles without rendering what is built) and the operator
dispatches them; ``fused`` puts several operators into one ELF. The pieces the operators share: ``design`` (aie.iron
helpers), ``mlir_util`` (walking a design's MLIR), ``l1_layout`` (pinned L1 buffers), ``kmath`` (math functions per
contract), ``paging`` (code pages), ``bfp16`` (host conversions), ``util``; the kernels' headers are
``artlite/include``.
"""

from . import design
from .base import ArgSpec
from .fused import FusedOperator

design.install_iron_patches()

__all__ = [
    "ArgSpec",
    "FusedOperator",
]
