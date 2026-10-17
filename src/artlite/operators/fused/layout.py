"""Where a fused ELF's named buffers live: three host spaces (input, output, scratch), each passed to its runtime
sequence as one argument, but scratch as several (segments) when it exceeds what one argument may span.

A runlist names its buffers per operator argument; ``"W[a:b]"`` names bytes a .. b of buffer W. A buffer is as large
as the largest argument that references it whole, or as given (``sizes``; required for a buffer referenced only by
slices). The input and output spaces hold the input and output buffers in the given order, scratch the others in
the order of their first reference, the external ones (living in another BO) last.
"""

import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import NamedTuple

from ..base import ArgSpec, ArtcOperator

# Bytes of scratch one runtime-sequence argument may span: a full ELF's relocations carry 32-bit addends (aiebu
# rejects larger ones), so a larger scratch space (a weight image and a 128K-position KV cache: > 10 GB) is passed as
# several arguments.
SEGMENT_BYTES = 0xF000_0000

Runlist = Sequence[tuple]  # (operator, *buffer names): one entry per operator run


class Span(NamedTuple):
    space: str  # "input", "output" or "scratch"
    offset: int  # bytes
    nbytes: int


def parse_name(name: str) -> tuple[str, int, int | None]:
    """A buffer reference -> (buffer, start, end): ``"W[a:b]"`` -> ("W", a, b), ``"W"`` -> ("W", 0, None)."""
    if not name.endswith("]"):
        return name, 0, None
    base, rng = name[:-1].split("[")
    start, end = rng.split(":")
    return base, int(start), int(end)


def spec_bytes(spec: ArgSpec) -> int:
    return math.prod(spec.shape) * spec.dtype.itemsize


class BufferLayout:
    def __init__(self, runlist: Runlist, *, inputs: Sequence[str], outputs: Sequence[str],
                 sizes: Mapping[str, int], external: Collection[str] = ()):
        whole: dict[str, int] = {}  # buffer -> bytes of its largest whole reference
        sliced: set[str] = set()
        for op, *names in runlist:
            specs: list[ArgSpec] = op.get_arg_spec()
            if len(specs) != len(names):
                raise ValueError(f"buffer count ({len(names)}) != arg spec count ({len(specs)}) for {op!r}")
            for spec, name in zip(specs, names):
                base, _, end = parse_name(name)
                if end is not None:
                    sliced.add(base)
                elif base not in whole or spec_bytes(spec) > whole[base]:
                    whole[base] = spec_bytes(spec)
        if unsized := sorted(b for b in sliced if b not in whole and b not in sizes):
            raise ValueError(f"buffers referenced only by slices need explicit sizes: {unsized}")
        self.spans: dict[str, Span] = {}
        self.space_bytes: dict[str, int] = {}
        own = [n for n in whole if n not in inputs and n not in outputs]
        own += [n for n in sizes if n not in inputs and n not in outputs and n not in whole]
        for space, names in (("input", inputs), ("output", outputs),
                             ("scratch", [n for n in own if n not in external] + [n for n in own if n in external])):
            offset = 0
            for name in names:
                nbytes = sizes.get(name, whole.get(name))
                if nbytes is not None:
                    self.spans[name] = Span(space, offset, nbytes)
                    offset += nbytes
            self.space_bytes[space] = offset

    def span(self, name: str) -> Span:
        """Where a buffer or a slice of one lies."""
        base, start, end = parse_name(name)
        s = self.spans[base]
        return s if end is None else Span(s.space, s.offset + start, end - start)


def scratch_segments(spans: Iterable[Span], limit: int) -> list[tuple[int, int]]:
    """[start, end) ranges of scratch, one runtime-sequence argument each, such that every scratch span lies inside
    one. Segments may overlap (a slice that crosses a segment's limit starts the next)."""
    segs: list[tuple[int, int]] = []
    for start, end in sorted({(s.offset, s.offset + s.nbytes) for s in spans if s.space == "scratch"}):
        assert end - start <= limit, f"a scratch buffer of {end - start} bytes exceeds one argument ({limit})"
        if segs and end <= segs[-1][0] + limit:
            segs[-1] = (segs[-1][0], max(segs[-1][1], end))
        else:
            segs.append((start, end))
    return segs or [(0, 0)]


def distinct(ops: Iterable[ArtcOperator]) -> list[ArtcOperator]:
    """The operator objects of ``ops`` once each, in the order of their first occurrence."""
    seen: dict[int, ArtcOperator] = {}
    for op in ops:
        seen.setdefault(id(op), op)
    return list(seen.values())
