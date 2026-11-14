"""The cores' roles: pages of code (``paging``) beyond the resident main loop of the vision overlay (``core``), what each
role's page enters, and which ops run it.

A role's page enters ``{p}_ext_<role>`` (the main loop calls it through the pager's state words), then its ``more``
entries (the GEMM's epilogues of that page: "ln" GEGLU second, GELU third; "d" SwiGLU second, the gated product third; "r"
SiLU second and third). An op whose commands run a role's code names it in ``seq.RowOps.role``; the sequencer puts the
role's page into the row's B stream where the row's cores hold another one.
"""

from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    """The roles by name (``core``'s docstring: what each one's code is)."""

    A = "a"  # attention of d 80
    C = "c"  # attention of d 40
    V = "v"  # the VAE decoder's single-head attention of d 512
    GN = "gn"  # the GroupNorm applies
    LN = "ln"  # the LayerNorm / RMSNorm apply, the GEGLU / GELU epilogues
    W = "w"  # MiniCPM-V's window attention
    R = "r"  # the RoPE epilogue, SiLU at GELU's place
    D = "d"  # depthwise convs, SwiGLU / gated-product epilogues
    S = "s"  # attention of d 160
    E = "e"  # LFM3-Audio's speech generator epilogues
    G = "g"  # the d-160 attention with per-query masks and the output gate


@dataclass(frozen=True)
class RoleSpec:
    """more: the page's entries after ``ext_<role>``; gemm_calls: the entries the resident GEMM calls through the state
    words; clobbers_table: the role's ops use the activation-table buffer (the rows' tables must be loaded again)."""

    more: tuple[str, ...] = ()
    gemm_calls: tuple[str, ...] = ()
    clobbers_table: bool = False


SPECS: dict[Role, RoleSpec] = {
    Role.A: RoleSpec(),
    Role.C: RoleSpec(clobbers_table=True),  # (d 40's scores in the table buffer)
    Role.V: RoleSpec(),
    Role.GN: RoleSpec(),
    Role.LN: RoleSpec(more=("geglu", "gelu"), gemm_calls=("geglu", "gelu")),
    Role.W: RoleSpec(),
    Role.R: RoleSpec(more=("silu_pn", "silu_pn"), gemm_calls=("ext_r", "silu_pn")),
    Role.D: RoleSpec(more=("swiglu", "prod"), gemm_calls=("swiglu", "prod")),
    Role.S: RoleSpec(),
    Role.E: RoleSpec(gemm_calls=("ext_e",)),
    Role.G: RoleSpec(),
}
ROLES = tuple(Role)
