"""A kernel variant and its rendering: the operator's kernel source with the variant's patches applied."""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .patch import apply_pragma, pragma_lines, replace_function, substitute


@dataclass
class Variant:
    """A named kernel variant: patches applied to the rendered source in this order: parameters (the operator's
    or the render function's keyword arguments), function definitions (name substring -> new definition),
    substitutions (old, new), the pragma on the target functions' loops (None: as rendered)."""

    name: str
    pragma: str | None = None
    subs: list[tuple[str, str]] = field(default_factory=list)
    defines: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    # the rolled loops (indices: the targets' in source order, then their always_inline helpers') that take the
    # pragma; None: the innermost ones
    loops: list[int] | None = None


@dataclass
class _Rendered:
    source: str  # the compiled file: the patched kernel source's translation unit (``kernel_source``)
    kernel_source: str  # the patched kernel source (what get_kernel_src returns)
    clang_cmd: list[str]
    pragma_lines: list[int]  # in the compiled file


def render_variant(
    factory: Callable[..., Any], kwargs: Mapping[str, Any], variant: Variant, *, functions: Sequence[str]
) -> _Rendered:
    """Render the operator (or the render function) with the variant's parameters and apply its patches. The patches
    see the rendered source's names (``{{ p }}_<name>``: ``kernel_source.MARK`` ...) and may also write them as the
    compiled file has them."""
    from ..operators.base import AIE_TARGET, CLANG_CMD_COMMON
    from ..operators.kernel_source import KernelSource

    made = factory(**{**kwargs, **variant.params})
    if isinstance(made, str):
        source, flags = made, []
    else:
        source, flags = made.get_kernel_src(), list(made.clang_flags)
    raw_names = {v: k for k, v in KernelSource(source).names.items()}

    def raw(text: str) -> str:  # (names as compiled -> as rendered)
        return re.sub(r"\b\w+_[0-9a-f]{8}\b", lambda m: raw_names.get(m.group(), m.group()), text)
    try:
        for name, definition in variant.defines.items():
            source = replace_function(source, name, raw(definition))
        for old, new in variant.subs:
            source = substitute(source, raw(old), raw(new))
        if variant.pragma is not None:
            source = apply_pragma(source, functions, variant.pragma, loops=variant.loops)
        unit = KernelSource(source).text
        lines = pragma_lines(unit, functions, loops=variant.loops) if variant.pragma is not None else []
    except ValueError as exc:
        raise ValueError(f"variant {variant.name}: {exc}") from None
    return _Rendered(unit, source, [*CLANG_CMD_COMMON, f"--target={AIE_TARGET}", *flags], lines)
