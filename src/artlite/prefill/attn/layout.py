"""The attention cores' L1: the workspace's regions, the bytes a core needs for a shape (``core_bytes``: what
``Shape.blocks`` fits), and the addresses of its buffers by 16 KB region (``l1_layout``)."""

from typing import TYPE_CHECKING

from ...operators.l1_layout import L1_BYTES, REGION
from ...operators.util import regions

if TYPE_CHECKING:
    from .attention import AttnOperator
    from .shape import Shape

STACK = 2048  # a core's stack
L1_LIMIT = L1_BYTES - 512  # a core's data memory for its buffers (a margin for aiecc's)
OS = REGION // 4  # (floats from o's first copy to its second: a region apart)


def workspace_regions(rows: int, cr: int, qb: int = 0, nc: int = 0) -> list[tuple[str, int]]:
    """The workspace's regions at ``rows`` row blocks and chunks of ``cr`` records (the kernels' comment); ``qb``: the
    packed variant's query blocks (their rows' segment starts); ``nc``: the non-causal variant's (their rows' keys)."""
    return [("sc", 256 * rows * cr), ("bias", 256 * rows), ("l", 256 * rows), ("vm", 192), ("st", 64)] + (
        [("lo", 128 * qb + 128)] if qb else []) + ([("lh", 32 * nc)] if nc else [])


def core_bytes(shape: "Shape", qb: int, cr: int) -> int:
    """L1 bytes of an attention core at qb row blocks per head and chunks of cr records (``AttnOperator``)."""
    rows = qb * shape.qh
    ws = regions(workspace_regions(rows, cr))["size"]
    return (STACK + 4 * shape.unit + 2 * qb * shape.nd * 72 + rows * shape.qunit + rows * shape.nd * 256
            + ws + 72 * rows * cr * shape.pp)


def two_copies(op: "AttnOperator") -> dict[str, list[int]] | None:
    """The layout of o in two copies (``l1_layout``), None where it does not fit."""
    s, R = op.shape, REGION
    rows, u = op.qb * s.qh, s.unit
    q, out, ws = rows * s.qunit, op.qb * s.nd * 72, op.workspace()["size"]
    r0 = STACK + q + 2 * out
    if op.o_bytes > R or r0 + op.pp_bytes > R or 3 * R + ws + 4 * u > L1_LIMIT:
        return None
    return {"la_q": [STACK], "la_or": [STACK + q, STACK + q + out], "la_pp": [r0], "la_wo": [R],
            "la_ws": [3 * R], "la_k": [3 * R + ws, 3 * R + ws + u], "la_v": [3 * R + ws + 2 * u, 3 * R + ws + 3 * u]}


def l1_layout(op: "AttnOperator") -> dict[str, list[int]]:
    """Core byte addresses by 16 KB region (the stack at 0). With o in two copies (``oc`` 2, where L1 holds them): the
    stack, Q, the output buffers and P in the first region, o's copies in the second and third (P V reads one and
    writes the other: a region each), the workspace and the K and V objects in the last; every hot loop's streams
    then lie in distinct regions (Q K: Q | K; the softmax: sc | P; P V: P | V | o | o'). Else the stack, Q and the
    output buffers (touched by ``fin`` only) in the first (Q K streams Q against K, whose objects sit with the
    workspace in the last), P there too where it fits, O from the second on (P V: O against P and V in stream B's
    objects after O)."""
    if op.oc == 2:
        return two_copies(op)
    s, R = op.shape, REGION
    rows, u = op.qb * s.qh, s.unit
    q, wo, ws, out = rows * s.qunit, 64 * s.nd * rows * 4, op.workspace()["size"], op.qb * s.nd * 72
    lay = {"la_q": [STACK], "la_or": [STACK + q, STACK + q + out], "la_wo": [R]}
    r0 = STACK + q + 2 * out
    if r0 + op.pp_bytes <= R:
        lay["la_pp"], pp = [r0], 0
    else:
        pp = op.pp_bytes  # (P after the workspace)
    pos = R + wo
    lay["la_v"], pos = [pos, pos + u], pos + 2 * u
    top = -(-pos // R) * R  # (the workspace from the next region on, stream A's objects after it)
    if top + ws + pp + 2 * u > L1_LIMIT:
        lay["la_k"], pos = [pos, pos + u], pos + 2 * u
        top = max(pos, L1_LIMIT - ws - pp)
        lay["la_ws"] = [top]
    else:
        lay["la_ws"], lay["la_k"] = [top], [top + ws + pp, top + ws + pp + u]
    if pp:
        lay["la_pp"] = [top + ws]
    assert STACK + q + 2 * out <= R and max(v[-1] + n for v, n in ((lay["la_ws"], ws + pp), (lay["la_k"], u))) <= L1_LIMIT, lay
    return lay
