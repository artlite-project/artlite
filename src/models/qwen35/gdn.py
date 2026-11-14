"""The GDN core of the multi-slot decode overlay (``qwen35/decode.py``): one Gated DeltaNet step of S <= 4 token
slots (chunkwise and 8 slots: below) per GDN layer and column, with a lazy commit of the recurrent state.

Column c holds q / k heads {2c, 2c + 1} and its HC v heads (``image.gdn_v_heads``; v head h of the column reads local
k head h & 1). The heads come in pairs (2 c + 16 p, + 1). Per step the core gets:

- the main core's message (the gdn_in phase's rows, stream M, ``C_LEN`` values per object; object j of slots 0 .. S - 1,
  then j + 1): per slot [raw qkv (CH = (4 + HC) 128 channels: q 2 heads, k 2 heads, v HC heads) | z (HC x 128) | alpha,
  beta (HC fp32 each)], all of them before any X object of the step (the main core skips those only after its phase);
- X objects (the others skip them: the first one's metadata, ``F_GDN`` with ``M_SKIP``), each ``XPAY`` bytes of
  payload: the conv objects of the q / k blocks (block b: 128 channels, [taps 4 x (w hi | w lo) bf16 | history 7 x 128
  bf16 | pad 256 B]), the K object of the previous step's tokens ([token 4][k head 2][k hi | k lo]), then per pair of v
  heads: its two conv objects, the delta object of the previous step ([head 2][token 4][128] fp32) and its 32 state
  panels (fp32, [16 j panels][16 i blocks][8 j][8 i] of a head's state);
- from the gdn_in metadata object: S, the tokens to replay (``M_NREP``), the conv shift (``M_CSHIFT``: the accepted tokens
  of the previous step) and the commit flag (``M_COMMIT``).

A conv object's history holds the 3 rows before the previous step's tokens and those tokens' raw rows: the step's base
rows are rows cshift .. cshift + 2. Its new history ([base 3 | the step's S rows]) and pad go out on D in the object's
place. The pads: block 0 [d of the step's tokens (the next step's replay) [token 4][8] fp32 | dt 8 | A 8 fp32], blocks
1, 2 ssm_norm (64 fp32 each), passed through.

Lazy commit: per panel the core replays the previous step's ``M_NREP`` accepted tokens (S <- S + d S + k delta^T from
the kept k, delta, d: the same arithmetic as their first pass), writes that base state out, then runs the step's S
tokens on it in place (their updates are not written: the next step replays the accepted ones). With ``M_COMMIT`` it
writes the state after the step's tokens instead (S = 1 decoding, before a prefill). Outputs on D (objects of
``D_BYTES``): the q / k blocks' histories (2 blocks per object), the step's K object, and per pair: its v blocks'
histories, the 32 state panels, the delta object of the step's tokens and the gated outputs rms(o) ssm_norm silu(z)
of the pair's heads, slot-interleaved as the gdn_out phase reads them ([head 2][k block 16][slot 4][8] bf16, then
2 KB of padding).

Chunkwise (``GdnShape(chunk=True)``: an MoE model with its drafter, ``overlay.chunk_of``, 4 slots; every design of 8 slots):
the delta rule in its WY / UT form (the prefill's ``gdn_prefill``: its docstring has the equations) in blocks of 8 positions
(position mod 8: the block's lanes), a step about one pass per panel whatever its S, each lane's numbers the same whatever the
steps that run it (a draft rejected or accepted, a step of one or of eight: the outputs of steps of S slots are those of steps
of one, bit for bit). The state is materialized only at a block's end: a step reads its block's base state and recomputes every lane of the
block up to its own (the lanes before it from the k / v histories and the gates of the block, [the 3 rows before the block
| its lanes' rows], which run on in the objects, read from row M_KSHIFT: 8 after a block's end); a lane's U by forward
substitution in lane order (``solve``), O from it and Y; the products along the lanes are per lane in fp32 (bfp16 blocks run
along i, a lane's own), so no lane's rounding depends on another's. The states are double-buffered (``GdnCursor.half``:
the half a position's block starts from): a step reads its block's half and writes the other, the block's update
(S + dc S + U'^T K^T over its 8 lanes) when it reaches the block's end, else nothing that is read (the other half is
read only after a block's end wrote it); a step past the block's end then runs
the next block's lanes from that update in L1 (their V in o's rows the outputs do not take, their K panels in k2). A
prefill ends anywhere: the decode's first block after it has no lanes before the prefill's end, whose zero gates leave the
state as it is, so that blocks start at multiples of 8 and a prefill (at a multiple of 32) at a block's start, the state
whole. The core runs per v head, its v heads grouped by k head (k head kh's v heads h = kh, kh + 2, ..), on the payload slots
of its groups (each object's metadata the 256 bytes before it, the previous payload's last; the first object's: F_GDN,
M_SKIP).

4 slots: a step's time at S 1 is its objects through DDR, so the core keeps the message on M in its workspace (no drains, no
sync after gdn_in): per k head [P, W, K], per v head [A, H]. P = [the q block's history (7 rows: [base 3 | the step's S rows], taken at cshift) | the block's lanes' gates
at P_D ([lane 12][d 8 | beta 8] fp32) | dt, A, ssm_norm at P_PRM (the first k head's)], W = [q weights | k weights], K = the
k block's history (16 rows), A = [the v block's weights | its history's rows 0 .. 7], H = [rows 8 .. 15 | 2 KB]. Per step,
after the message: per k head P (the gates), W (its q block at the step's lanes, Q's panels), the P object out, K (its k
block over the block's lanes: K's panels, the Gram blocks, K^T's panels), the K object out, then per v head A, H (its V, the
lanes' constants: decays, L and M's columns, ``vhead``), 16 state panels and out: 16 state panels, [its gated outputs (1 KB)
| 1 KB | its history's rows 0 .. 7], [rows 8 .. 15 | 2 KB]. Conv outputs come out in panel layout ([panel 16][lane 8][8]:
``pconv``). Program memory is what bounds this core: cold functions are ``minsize`` where that measured smaller, vectors go by
value through no call (a 64-lane argument costs more at its call sites than an inlined product).

8 slots (``GdnShape(slots=8)``): the message of 8 slots (34 KB) does not fit the core's L1 (which also holds the helper's
scratch: ``decode.design``'s 8-slot partition), so gdn_in's rows reach it through DDR: gdn_in drains them (C) into the step's
objects (3 drains per column, a block's rows (half, slot) as [half][slot 8][64]: the q blocks' and the K objects, the R
objects' v rows then z, ab), then a sync. The header [the slots' ab | dt, A, ssm_norm | the block's lanes' gates (16 lanes,
from the last gated outputs' D object)], per k head [q A, q B, k A, k B, K], per v head [A, B, R]: a conv block's A = [weights
(2 KB) | its history's rows 0 .. 7], B = [rows 8 .. | the step's q rows] (q: [base 3 | the step's S rows], taken at cshift) or
[rows 8 .. 23] (a k or v block's history, block-linear as the 4-slot core's, 24 rows: 3 + 7 lanes before a step + 8 of it +
6 past them), K = the step's k rows, R = [the step's v rows | z]. The step's words come from the gdn_in phase's metadata object
(``F_GIN``). Per step: the header (``gbegin``: the gates), per k head its q block (its history out), its k block and K object
(K's panels; its history out), then per v head A, B, R (its V), 16 state panels, its history and gated outputs out ([k block
16][slot 8][8] bf16: gdn_out's x of the head; the 2 KB after them the block's gates, into the header from the last head, else
the sink). The helper core (``hgdn``: it reaches the core's memory and is idle in a GDN step; asked by the core's lock
``L_HRDY``, done on ``L_HDONE``; the header's ``M_HPAN`` tells it the requests of a step through a K_HELP command of the main
core) computes per k head the Gram blocks and K^T's panels (after its K object; the first v head's R object waits for them),
per v head its chunks' constants (``vhead``, after its R object; the panels wait for them).

Branch (8 slots, ``QwenOverlay.step``'s branch; ``M_BRANCH`` in gdn_in's metadata): the step's last L slots follow slot c
instead of the chain's last, at positions inside slot c's block. The chain is the step's other slots (S counts them), the
branch a third chunk from the block's base over the lanes through its last: the chain's to slot c and its own (its conv
windows: slot c's rows, then its own; its K and Q panels: the chain's with its lanes', its gates after the chain's in dn's
lanes 16 ..), each lane as in a step of one at its position. Its regions lie in the helper's scratch (``HS8_AT``, pinned:
idle while the helper waits in ``hgdn``, which the step's last v head releases only after its panels: the helper's next phase
starts as the first heads' outputs come in), its outputs' lanes go into o's lanes past the chain's, where its slots' outputs
read them.
"""

import math
from typing import Self

import numpy as np
from jinja2 import Template

from artlite.decode.protocol import PHASE, SLOTS, XMETA
from artlite.model_config import QwenConfig
from artlite.operators import templating
from artlite.operators.bfp16 import PANEL_BYTES
from artlite.operators.l1_layout import REGION
from artlite.operators.util import Attr, bf16_hilo, bf16_hilo_bits, mask64, regions
from artlite.operators.words import WordMap

from .gdn_math import MATH, PANEL, D, GdnDims
from .words import FLAGS, GDN_IN

D_BYTES = 4096  # bytes of a D object
HIST = 3 + SLOTS  # history rows of a conv object

# state words of the GDN core's workspace (the first ones are trip counts of its program): one array of the three
# layouts (4 slots, chunkwise, 8 slots), which read some words as other things (the aliases)
STATE = WordMap("GDN core state", 19)
_G = STATE.common
(G_NSKIP, G_NMSG, G_ON, G_NCP, G_KO, G_PO, G_NS, G_NREP, G_CSHIFT, G_COMMIT, G_MJ, G_MS, G_CB, G_PAIR,
 G_PANEL) = _G.words("G_NSKIP G_NMSG G_ON G_NCP G_KO G_PO G_NS G_NREP G_CSHIFT G_COMMIT G_MJ G_MS G_CB G_PAIR G_PANEL", at=0)
G_KH, G_VI, G_VH = _G.alias("G_KH", G_PAIR), _G.alias("G_VI", G_MJ), _G.alias("G_VH", G_MS)  # (chunkwise) the k head, its v heads done, the v head in the column
G_HK = _G.alias("G_HK", G_KO)  # (8 slots) the helper's next work: a k head's Gram blocks (1), a v head's constants (0)
G_KSHIFT = _G.alias("G_KSHIFT", G_PO)  # (chunkwise) the k / v histories' base row (M_KSHIFT)
G_NPD, G_NPX = _G.alias("G_NPD", G_NCP), _G.alias("G_NPX", G_COMMIT)  # (8 slots) a v head's state panels with a D object (a step
# that ends its block: its update out), without one (else: nothing of them is read)
G_BR = _G.word("G_BR")  # (8 slots) a branching step's word (``M_BRANCH``: its parent slot + 1 | its slots << 8, or 0)
G_WORDS = G_BR + 1
G_TA = _G.block("G_TA", 2)  # (chunkwise) the SiLU table's addresses
G_BN = _G.word("G_BN")  # (8 slots) a branching step's branch chunk: the block's lanes through its last (0: none)


def consts() -> dict[str, int]:
    """The words and flags the GDN core's kernels take: the metadata words of its objects and its state words."""
    return XMETA.consts(PHASE, GDN_IN, FLAGS) | STATE.consts()


# (8 slots) the GDN tile's locks of the k and v heads' constants, which the helper core computes (``hgdn``): asked, done;
# the SiLU table's two copies in the core's regions 0, 1 (bytes); the header's metadata word of the helper's runs per step (its
# heads' and the step's end): ``M_HPAN``
L_HRDY, L_HDONE = 12, 13
SILU_TAB = (0x739C0, 0x7F9C0)
# (8 slots) the helper's scratch on the GDN tile (local address, pinned: the helper waits in ``hgdn`` until the step's end; the
# cores address it from the workspace, the helper's view of the tile differs from the GDN core's):
# a branching step's branch there, V and O ([panel 16][lane 8][8] fp32) at HB_V, K and Q panels at HB_K, HB_Q, its rows'
# scratch (the 3 rows before it, its own) at HB_R
GWS_AT = REGION  # (chunkwise) the GDN workspace's local address (pinned: regions 1 .. 3, below the helper's scratch (8 slots)
# and a SiLU table copy)
HS8_AT = 0xD1C0
HB_V, HB_K, HB_Q, HB_R = 0, 4096, 4096 + 16 * 2 * PANEL_BYTES, 4096 + 32 * 2 * PANEL_BYTES
# (8 slots) a column's objects of a GDN layer: payload slots of XPAY bytes (each object's metadata the 256 bytes before
# its payload): the header (H_SLOT), per k head kh its group [q A, q B, k A, k B, K] (from KH_SLOT + KH_SPAN kh), per v
# head h its group [A, B, R] (from V_SLOT + V_SPAN h); the header's payload: the slots' ab at H_AB (a gdn_in object
# each: 128 bytes), dt 16 | A 16 | ssm_norm 128 fp32 at H_PRM, the block's lanes' gates at H_DN ([lane 16][d 8 | beta 8]
# fp32: the last gated outputs' D object's second half)
H_SLOT, KH_SLOT, KH_SPAN, V_SLOT, V_SPAN = 0, 1, 5, 11, 3
H_AB, H_PRM, H_DN = 0, 1024, 2048
# (chunkwise, 4 slots) a column's objects of a GDN layer: per k head its group [P, W, K], per v head its [A, H]; a P
# object's bytes: its q block's history, the block's lanes' gates at P_D ([lane 12][d 8 | beta 8] fp32), the ssm parameters
# at P_PRM; a K object: its k block's history (16 rows); A: the v block's weights, its history's rows 0 .. 7 at 2048; H: the
# history's rows 8 .. 15
KH_SPAN4, V_SLOT4, V_SPAN4, P_D, P_PRM = 3, 6, 2, 1792, 2560


_SRC = templating.load("models/qwen35/gdn.cc.jinja")


# The chunkwise cores (module docstring "Chunkwise"): their pieces in order, one template (the first one's {% set %} names
# serve the others)
def _chunkwise(entry: str, out: str) -> Template:
    parts = ("ca", "l2p4", "cn4", "gram4", "cb4", entry, "gpanel4", out)
    return templating.JINJA_ENV.from_string("".join(templating.source(f"models/qwen35/gdn.{n}.cc.jinja") for n in parts))


_SRC8 = _chunkwise("entry8", "out8")
_SRC4 = _chunkwise("entry4", "out4")


class GdnShape(GdnDims):
    """The GDN core's sizes for a model: HC v heads and CH conv channels per column (``GdnDims``), the message, the
    workspace; the core of steps of up to ``slots`` (4: the message on M; 8: chunkwise, module docstring "8 slots"), of
    4 slots ``chunk``wise (module docstring "Chunkwise"). ``GdnShape(cfg, slots=, chunk=)`` is the shape of its role:
    ``Gdn4``, ``Gdn4Chunk`` or ``Gdn8``, which hold that role's object layout, workspace and source."""

    slots: int
    chunk: bool

    def __new__(cls, cfg: QwenConfig | None = None, /, *, slots: int = SLOTS, chunk: bool = False,
                c_len: int = 64, xpay: int = 4096, meta: int = 64) -> Self:
        if cls is GdnShape:
            assert slots in (SLOTS, 8)
            return super().__new__(Gdn8 if slots == 8 else Gdn4Chunk if chunk else Gdn4)
        return super().__new__(cls)

    def __init__(self, cfg: QwenConfig, *, c_len: int = 64, xpay: int = 4096, meta: int = 64, slots: int | None = None,
                 chunk: bool | None = None):
        assert slots in (None, self.slots) and (chunk is None or (chunk or self.slots == 8) == self.chunk)
        super().__init__(cfg)
        self.c_len, self.xpay, self.meta = c_len, xpay, meta
        self.msg_values = self.ch + self.hc * D + 4 * self.hc  # (alpha, beta as fp32: 2 bf16 each)
        self.n_msg = -(-self.msg_values // c_len)
        self.msgp = self.n_msg * c_len  # bf16 elements of a slot's message
        assert 8 * D * 2 + HIST * D * 2 + 4 * SLOTS * 16 <= xpay  # a conv object
        assert SLOTS * 2 * 2 * D * 2 <= xpay and 2 * SLOTS * D * 4 <= xpay  # the K and delta objects


class Gdn4(GdnShape):
    """The core of 4 slots whose message stays on M (module docstring "4 slots" of the decode core)."""

    slots, chunk = SLOTS, False

    @property
    def n_objects(self) -> int:
        """X objects of a step (after the gdn_in phase's): conv q / k 4, K, per pair 2 conv + delta + 32 panels."""
        return 5 + self.hc // 2 * 35

    def workspace(self) -> dict[str, int]:
        sizes = [("st", 4 * G_WORDS), ("prm", 4 * 160), ("msg", 2 * SLOTS * self.msgp), ("kq", 2 * SLOTS * 8 * D),
                 ("gt", 4 * SLOTS * 48), ("v", 4 * 2 * SLOTS * D), ("o", 4 * 2 * SLOTS * D), ("dl", 4 * 2 * SLOTS * D),
                 ("kold", 2 * SLOTS * 4 * D), ("dold", 4 * SLOTS * 16), ("y", max(4 * D, 2 * SLOTS * D)), ("uw", 4 * 16)]
        return regions(sizes)

    def render(self, p: str) -> str:
        return _SRC.render(p=p, g=Attr(self.workspace()), gm=MATH, sigmoid=False, CH=self.ch, HC=self.hc, D=D, n_msg=self.n_msg,
                           MSGP=self.msgp, C_LEN=self.c_len, META=self.meta, XPAY=self.xpay, D_BYTES=D_BYTES,
                           PANEL=PANEL, SMAX=SLOTS, HIST=HIST, eps=repr(self.eps),
                           rs=bf16_hilo(1 / math.sqrt(D)), **consts())


class Gdn4Chunk(GdnShape):
    """The chunkwise core of 4 slots (module docstring "Chunkwise"): the objects per k head [P, W, K], per v head [A, H]."""

    slots, chunk = SLOTS, True

    @property
    def n_objects(self) -> int:
        """X objects of a step (after the gdn_in phase's)."""
        return V_SLOT4 + self.hc * (V_SPAN4 + 16)

    @property
    def n_slots(self) -> int:
        """Payload slots of a column's objects of a layer."""
        return self.v_slot(self.hc)

    def kh_slot(self, kh: int) -> int:
        """The first payload slot of k head kh's group [P, W, K]."""
        return KH_SPAN4 * kh

    def v_slot(self, h: int) -> int:
        """The first payload slot of v head h's group [A, H]."""
        return V_SLOT4 + V_SPAN4 * h

    def weights_at(self, b: int) -> int:
        """Bytes from a column's first payload to conv block b's weights."""
        if b >= 4:
            return self.xpay * self.v_slot(b - 4)
        return self.xpay * (self.kh_slot(b % 2) + 1) + (2048 if b >= 2 else 0)

    def hist_at(self, b: int) -> int:
        """Bytes from a column's first payload to conv block b's history."""
        if b >= 4:
            return self.xpay * self.v_slot(b - 4) + 2048
        return self.xpay * (self.kh_slot(b % 2) + (2 if b >= 2 else 0))

    def workspace(self) -> dict[str, int]:
        S = self.slots
        # (kq, qp, k2: the block's K, the step's Q (at their lanes), the next block's K panels ([ib 16][hi, lo]); gg: the
        # Gram blocks of the two blocks; cc: per chunk its constants [L's, M's columns | beta | Gamma - 1]; xy: also the
        # scratch of the norms, the gates' rows (vhead) and U' (uprep); kh: a k head's k history, then a v head's;
        # hb | dn | prm: a P object's in a row, its D object one call)
        assert P_D == 2 * (3 + S) * D and P_PRM - P_D == 64 * 12
        pan2 = 16 * 2 * PANEL_BYTES
        sizes = [("st", 4 * (G_WORDS + 2)), ("msg", 2 * S * self.msgp), ("kq", pan2), ("qp", pan2),
                 ("k2", pan2), ("kt", pan2), ("gg", 4 * 256), ("vt", 4 * 8 * D), ("o", 4 * 8 * D),
                 ("u", 4 * 64), ("er", 4 * 128 + 64), ("cc", 2 * 4 * 144), ("uf", 4 * 16), ("dc", 2 * 64),
                 ("xy", 4 * 128), ("cw", 2 * 8 * D), ("zb", 2 * 8 * D), ("kh", 4096), ("hb", P_D),
                 ("dn", P_PRM - P_D), ("prm", 4 * 160)]
        out = regions(sizes)
        assert out["prm"] == out["hb"] + P_PRM
        return out

    def render(self, p: str) -> str:
        strict, lower = mask64(lambda t, j: j < t), mask64(lambda t, j: j <= t)
        return _SRC4.render(p=p, g=Attr(self.workspace()), gm=MATH, HC=self.hc, CH=self.ch, D=D, C_LEN=self.c_len, META=self.meta,
                            MSGP=self.msgp, n_msg=self.n_msg, SMAX=self.slots, eps=repr(self.eps), rs=bf16_hilo(1 / math.sqrt(D)),
                            PAN=PANEL_BYTES, P_D=P_D, P_PRM=P_PRM, strict=f"{strict:016X}", lower=f"{lower:016X}", TAB=SILU_TAB,
                            **consts())


class Gdn8(GdnShape):
    """The chunkwise core of 8 slots (module docstring "8 slots"): the header, per k head [q A, q B, k A, k B, K], per v
    head [A, B, R]."""

    slots, chunk = 8, True

    def __init__(self, cfg: QwenConfig, **kwargs: int):
        super().__init__(cfg, **kwargs)
        # (the gdn_in objects of a slot: qkv rows, z, ab; the ab object at a C object's start)
        assert (self.ch + D * self.hc) % self.c_len == 0 and 4 * self.hc <= self.c_len

    @property
    def n_objects(self) -> int:
        """X objects of a step (after the gdn_in phase's): the header, per k head 5, per v head [A, B, R] + 16 panels."""
        return 1 + 2 * KH_SPAN + self.hc * (V_SPAN + 16)

    @property
    def n_slots(self) -> int:
        """Payload slots of a column's objects of a layer."""
        return self.v_slot(self.hc)

    def kh_slot(self, kh: int) -> int:
        """The first payload slot of k head kh's group [q A, q B, k A, k B, K]."""
        return KH_SLOT + KH_SPAN * kh

    def v_slot(self, h: int) -> int:
        """The first payload slot of v head h's group [A, B, R]."""
        return V_SLOT + V_SPAN * h

    def block_slot(self, b: int) -> int:
        """The A object's payload slot of conv block b (q heads 0, 1, k heads 0, 1, then the v heads)."""
        return self.kh_slot(b % 2) + 2 * (b // 2) if b < 4 else self.v_slot(b - 4)

    def weights_at(self, b: int) -> int:
        """Bytes from a column's first payload to conv block b's weights."""
        return self.xpay * self.block_slot(b)

    def hist_at(self, b: int) -> int:
        """Bytes from a column's first payload to conv block b's history."""
        return self.xpay * self.block_slot(b) + 2048

    def workspace(self) -> dict[str, int]:
        # (the 4-slot chunkwise core's regions but the message; dn: the block's gates of 16 lanes, then a branch chunk's 8;
        # gg, cc: a branch chunk's third; a branch's own regions in the helper's scratch, HS8_AT; vh: a k or v block's
        # history [the 3 rows before the block | its lanes' rows] (24 rows), whose last D object runs 2 KB past it (zr: the v
        # head's z rows, [half 2][slot 8][64] bf16); hb: the q history [base 3 | the step's 8 rows] in vh's first rows (out
        # before the k block's history comes in), a D object's 4 KB from it on)
        pan2 = 16 * 2 * PANEL_BYTES
        sizes = [("st", 4 * (G_BN + 1)), ("prm", 4 * 160), ("dn", 4 * 24 * 16), ("kq", pan2), ("qp", pan2),
                 ("k2", pan2), ("kt", pan2), ("gg", 4 * 384), ("vt", 4 * 8 * D), ("o", 4 * 8 * D),
                 ("u", 4 * 64), ("er", 4 * 128 + 64), ("cc", 3 * 4 * 144), ("uf", 4 * 16), ("dc", 2 * 64),
                 ("xy", 4 * 128), ("cw", 2 * 8 * D), ("zb", 2 * 8 * D), ("vh", 2 * 24 * D), ("zr", 2 * 8 * D)]
        out = regions(sizes)
        out["hb"] = out["vh"]
        assert out["vh"] + 2 * 24 * D + D_BYTES // 2 <= out["size"], out
        return out

    def render(self, p: str) -> str:
        strict, lower = mask64(lambda t, j: j < t), mask64(lambda t, j: j <= t)
        return _SRC8.render(p=p, g=Attr(self.workspace()), gm=MATH, HC=self.hc, D=D, C_LEN=self.c_len, META=self.meta,
                            SMAX=self.slots, eps=repr(self.eps), rs=bf16_hilo(1 / math.sqrt(D)),
                            H_AB=H_AB, H_PRM=H_PRM, H_DN=H_DN, HS_OFF=HS8_AT - GWS_AT, HB_V=HB_V, HB_K=HB_K, HB_Q=HB_Q, HB_R=HB_R,
                            PAN=PANEL_BYTES, strict=f"{strict:016X}", lower=f"{lower:016X}", L_HRDY=L_HRDY, L_HDONE=L_HDONE,
                            TAB=SILU_TAB, **consts())


# ----------------------------------------------------------------------
# Host side: object layouts
# ----------------------------------------------------------------------


def conv_weights(w: np.ndarray) -> np.ndarray:
    """(8 slots) The conv blocks' weights (CH, 4) fp32 -> per block its A object's first 2 KB: [tap 4][hi 128 | lo 128]
    bf16 (NB, 2048) uint8."""
    hi, lo = bf16_hilo_bits(np.asarray(w, np.float32).T)  # (4, CH)
    nb = hi.shape[1] // D
    wb = np.stack([hi.reshape(4, nb, D), lo.reshape(4, nb, D)], 2).transpose(1, 0, 2, 3)  # (NB, 4, 2, 128)
    return np.ascontiguousarray(wb).view(np.uint8).reshape(nb, -1)


def head_payload(dt: np.ndarray, ssm_a: np.ndarray, norm: np.ndarray) -> np.ndarray:
    """(8 slots) The header's static part (at H_PRM): dt, A of the column's heads (16 fp32 each), ssm_norm (128 fp32)."""
    out = np.zeros(160, np.float32)
    out[:len(dt)], out[16:16 + len(ssm_a)], out[32:] = dt, ssm_a, norm
    return out.view(np.uint8)


def conv_payloads(sh: GdnShape, w: np.ndarray, dt: np.ndarray, ssm_a: np.ndarray, norm: np.ndarray) -> np.ndarray:
    """The column's conv objects' payloads (NB, XPAY) uint8 at load: the weights (CH, 4) fp32 as [tap 4][hi 128 | lo 128]
    bf16, a zero history, the pads (block 0: d zero, dt, A of the column's heads; blocks 1, 2: ssm_norm)."""
    hi, lo = bf16_hilo_bits(np.asarray(w, np.float32).T)  # (4, CH)
    out = np.zeros((sh.nb, sh.xpay), np.uint8)
    pad0 = 2 * (8 * D + HIST * D)
    for b in range(sh.nb):
        wb = np.stack([hi[:, b * D:(b + 1) * D], lo[:, b * D:(b + 1) * D]], 1)  # (4, 2, 128)
        out[b, :wb.nbytes] = np.ascontiguousarray(wb).view(np.uint8).reshape(-1)
    pad = np.zeros(64, np.float32)
    pad[8 * SLOTS:8 * SLOTS + sh.hc], pad[8 * SLOTS + 8:8 * SLOTS + 8 + sh.hc] = dt, ssm_a
    out[0, pad0:pad0 + 256] = pad.view(np.uint8)
    nb = np.asarray(norm, np.float32)
    out[1, pad0:pad0 + 256], out[2, pad0:pad0 + 256] = nb[:64].view(np.uint8), nb[64:].view(np.uint8)
    return out
