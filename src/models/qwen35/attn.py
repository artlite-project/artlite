"""The attention core of the multi-slot decode overlay (``qwen35/decode.py``): the full-attention layers (head_dim 256,
NeoX RoPE on the first 64 dims, q / k RMS-normed per head, a sigmoid output gate) for the token slots of a step at
positions p .. p + S - 1.

The KV heads' columns form chains of NPH = 8 / KV heads (9B / 27B NPH 2, MiniCPM-V 4): column c attends with its QC
query heads (27B 3, 9B 2, MiniCPM-V 1) to KV head c // NPH over the whole cache (``decode.kv``'s d-block-major layout with
one K panel per 8 dims: groups of 32 positions, a K object of pairs of records [d-block][record 2] and a V object of pairs
of records [d-block][record 2]). Two cores, by design:

One pass (a design of 8 slots, ``AttnShape.one``; source ``_ONE``): the rows are a head's slots: octet h holds query head h's
8 slots as rows of one MMUL panel, single bfp16 Q and P, whatever the step's S (rows past S zero): a step's compute does
not follow S, it only has to stay below the stream's time. The chain reads the cache once: the first column's A stream
reaches every core of a KV head's chain. Per step a core gets

- from the X stream (the others skip them: ``M_SKIP``): the ``F_ATT`` object [q_norm x log2(e) / 16, k_norm (256 fp32 each)
  | cos, sin per slot (8 x 64 fp32)], the slots' q objects ([q | gate] x QC heads, bf16, as attn_in drained them), then the
  slots' raw K, V objects ([K 256 | V 256] per KV head, bf16);
- from the A stream (broadcast from the main core's, whose objects it skips; two buffers: an object streams in while
  the core works on the last one): the cache up to the group of the step's last position, then the two tail objects.

The slots' K rows (normed, rotated) wait as one panel per d-block by position & 7, a step's at most 8 positions apart there
(``kr``, ``kn``); each open record takes its rows as its K object passes (``kmerge``: lanes of the popped panels, in the core's
copy, the last core's and the first's own), and a K object scores both pairs of records against the octets, each K panel
popped once (``qk1``). Each row's positions past its own are masked (the host's masks of the step's first two records,
``M_AMASK``). A branching step (``M_BRANCH``, ``QwenOverlay.step``'s branch: its last slots follow an earlier slot c, inside
the 8 positions of c's record) scores that record again with the branch's K at its positions' lanes for the branch rows
(``bqk``), and puts the branch's V rows there for its slots' P V (``tobj``): a branch row sees its record as a step of one at
its position does. P = 2^(s - m), m the row's maximum in the first group,
held for the whole pass (scores span less than P's 2^63.5: no rescale; QK starts from the baseline 192 - m), y += P V over the V object's records. A step may cross a group: the first group's open record keeps its scores in ``sc`` past the
second group's (its slot is the group's last, the second group's open record its first). The tail objects are the open
records' rows (the ring: ``decode.vring``; after a prefill the host decodes the record's V panels into it,
``attn_steps.AttentionSteps``'s ``_rings_from_prefill``): the core puts the slots' V in and per slot encodes the record's V panels up to the
slot's position and adds the P V of its rows (a V block spans the record's positions: what a position sees of its record
does not depend on the step's later slots).

The outputs go to the main core (shared-memory FIFO R, ``R_LEN`` values per object), which applies the gates (``_MAIN_ONE``: the
core's L1 holds none) and relays them to C: o ([head][k block][slot 8][8]: an octet's rows as they lie), then a chain's first
core its merged K records of the step's first and last positions, its last core their V panels and rows (the same record
twice when one holds both). The two emit at once: neither waits for an object of the other.

L1: 3 octets take 25 KB of y and l, 3 KB of scores, 7 KB of Q panels; what the core reads rarely lies in the main core's
memory (``AttnShape.ext``), which the core reads through shared memory: the main core's workspace past its residual (``EXT_W8`` bytes:
GEMV scratch, idle while the attention runs; its next phase starts after a sync on the attention's outputs) and its sum
panels (``EXT_X8``).

Sub-steps (designs of 4 slots; source ``_SRC``): a step's attention runs in sub-steps (``AttnShape.sub`` slots of one group
at most, ``decode.vring.substeps``) whose S x QC query rows (row s QC + h: slot s, head h) go 8 to an octet, one MMUL panel of
single bfp16 per 8 dims, 4 to a set in y and the scores. Per sub-step the core gets from X the ``F_ATT`` object [q_norm x log2(e) / 16, k_norm | the
sub-step's cos, sin], q and raw K, V objects as above, and from A the cache of its head up to the sub-step's group, then (a core that
attends to that group) the tail objects of its records.

It inserts the slots' K into their records (in its copy), scores every record against each octet's two sets at once (log2 units:
Q x log2(e) / 16; per d-block the pair's K panels and the octet's Q panels each popped once), masks each row's positions past
its own, softmax with a running maximum per row (masked positions add exactly nothing; y and l rescaled once per group, a
row's m where it rises: a chat's first keys are no reference for the later ones), y += P V over octets (single bfp16 P, a V object's four records
in one read and write of y: ``pv4``). A group's softmax runs after its K object (``asoft``). The open records' values come from the ring of
the layer's KV head (``decode.vring``; after a prefill the host decodes the record's V panels into it) as the tail objects,
after the cache; the core puts the slots' V in and per slot encodes the record up to the slot's position and adds the P V
of its rows (what a position sees of its record does not depend on the sub-step's later slots).
Its outputs go to the main core (shared-memory FIFO R, ``C_LEN`` values per object), which relays them
to C: o = y / l * sigmoid(gate) ([head][k block: 8 dims][slot S][8] bf16: into the sub-step's slots of the attn_out phase's input),
then the records of the sub-step's first and last positions (of one slot: its record): a chain's first column the K object
of their group's pairs
that hold them, its last one their V panels up to the last position and their rows (the ring), the columns between none
(``A_ROLE``). The core holds its last two A objects until then and reads the records from them (``HOLD``): the tail objects,
or the sub-step group's K and V objects (a chain's first column gets them last: split, only those of that group; else again
after its tails).

A sub-step from 4 groups on (``AttnShape.splits``: the same for every position of its group) reads each head's cache
once per chain instead of once per column: every core of a chain takes the chain's NPH QC heads (the chain's q objects
in column order on X) and a run of the groups (``AttnShape.bounds``): the first core the groups 0 .. ``M_GATT`` - 1
(then the sub-step group's K and V objects from ``M_GJ`` on only for its K records), the last one the groups up to the
sub-step's. The partials (m, l, y) merge west to east: a core's partial stays in its workspace until its east
neighbour (shared memory) has merged it into its own, and the last core emits o for the chain's heads (a token
ObjectFifo orders each neighbour pair).

A sub-step of more slots than a chain of 4 holds (``AttnShape.rounds``: the 35B-A3B's 3 and 4 slots, 32 rows at most against
a core's 16) runs in two rounds, one per KV head (``SPLIT_ROUND``): in round r both chains take KV head r, the first the
heads of the head's first two columns, the second those of the other two, each core the run of the groups its place in a
chain has (the first chain's shims read the runs of either head on fills whose packet id also reaches the memtile of the
column 4 away, ``decode.design``'s "Shared A objects", which takes them as if its own shim had read them), and only KV
head r's own chain emits its records (``M_SPLIT``'s bits). A slot's row thus sees the groups in the same runs, merged in the
same order, in a sub-step of any slots. A round's second half follows its first on the same streams (no sync between them;
the token that orders a neighbour pair's merge also guards the y a core zeroes for its next sub-step: ``aq``). A stream per
column reads ~56 GB/s (8 at a time), two per KV head ~46, one for a whole chain ~28 (a chain of 4 in the one-pass form: S 1
+30% at 32K on the 1B). The chain's halves for every sub-step (two runs, two streams per KV head) cost the 35B-A3B's S 1 and
2 +1.6 ms at 32K (46.8 -> 48.3 ms at S 1); two passes of the chains over the cache (sub-steps of 2 slots) cost its S 3 and
4 ~7-9 ms.

L1: a sub-step's rows need per set ~5.3 KB (y and l fp32, scores, references, P panels) plus its Q panels (2.3 KB) and the
rows' gates (0.5 KB a row); the sigmoid's table takes 2.1 KB (``gate_tab``). Regions the core reads rarely (the gates, the
step's V, ...: ``AttnShape.ext``) lie in the main core's memory: its workspace past its residual (``EXT_W`` bytes), its sum
panels (``mx_bytes``), idle in a sub-step.
"""

from typing import Literal, NamedTuple

import numpy as np

from artlite.decode.kv import Records
from artlite.decode.protocol import META, PHASE, SLOTS, TOK_STATE, XB, XMETA
from artlite.decode.protocol import STATE as MAIN_STATE
from artlite.model_config import QwenConfig
from artlite.operators import templating
from artlite.operators.bfp16 import PANEL_BYTES, PANEL_ELEMS
from artlite.operators.design import COLS
from artlite.operators.kmath import SIGMOID
from artlite.operators.l1_layout import L1_BYTES
from artlite.operators.util import Attr, cos_sin_rows, regions, rup
from artlite.operators.words import WordMap

from .words import ATT, FLAGS

HD = 256  # head_dim
ND = HD // 8  # d-blocks (panels per record part)
REC = Records(HD, 1)
REC_K, REC_V, ROWS_B = REC.k_bytes, REC.v_bytes, REC.rows_bytes  # bytes of a K record, a V record, its rows (a ring slot)
POS_PER_GROUP = 32  # 4 records: a K object and a V object
OBJS = REC.objects  # cache objects of a group
OBJ_BYTES = 9280  # the overlay's objects: a weight tile (1024 columns: 16 rows of Q4_0, Q4_K) and a cache object
ROT = 32  # rotated pairs (dims i, i + 32 for i < 32)
MAX_ROWS = 16  # query rows of a sub-step at most (``AttnShape.sub``), the gates in the core's sight
MAX_ROWS_G = 24  # with the main core applying the gates (``AttnShape.main_gates``: dense models' designs)
# a V record's drain into its pair (txn.shim_bd dims, words): ND panels 144 B apart
V_SCATTER = (18, ND, PANEL_ELEMS, PANEL_ELEMS * ND)
K_SCATTER = V_SCATTER  # (one pass) a K record's drain into its pair
TAILS = 2  # tail objects of a sub-step that attends to its group: the ring slots of its first and last records
HOLD = 2  # the last A objects of an attention sub-step, held until its emission
STACK = 1600  # the core's stack (its deepest kernel call: ~1.2 KB)
EXT_W = 9728  # bytes of the main core's workspace past its residual (``decode.design``: inv .. scr), idle in a sub-step
EXT_X = 5760  # bytes of a dense design's main core's sum panels (``decode.design.xs_bytes``), idle in a sub-step
EXT_W8, EXT_X8 = 12288, 9600  # (the same of a design of 8 slots)
PO = 288  # bytes of an octet's P panels of a group (4 panels)
# state words of the main core for the gates (main_gates: its workspace's words past decode.protocol's of a dense design,
# where the token objects' words are not: ``GATES``): the q objects to gate, those seen, the slots, the query rows per slot,
# the R objects of o, those relayed
GATES = MAIN_STATE.kind("qwen35 gates", exclusive_with=(TOK_STATE,))
MG_Q, MG_I, MG_S, MG_QR, MG_NO, MG_E, MG_SI = GATES.words("MG_Q MG_I MG_S MG_QR MG_NO MG_E MG_SI", at=56)
# M_SPLIT, bits 1:0: the chain divides the cache (0: each column over all of it), a round's chains do; bit 2 (a round):
# its KV head; bit 3: no records (a round's other chain)
SPLIT_CHAIN, SPLIT_ROUND, SPLIT_KV, SPLIT_NOREC = 1, 2, 4, 8


# metadata words of an F_ATT object (``words.ATT``): the protocol's M_NS (slots), M_SKIP (X objects after it), M_NASKIP (the
# cache's A objects and the tail objects), M_NREL (R objects to relay), M_POS (the first slot's position), M_SA (the first
# slot's row of angles in the object: 0, each sub-step's object holds its own slots'), M_NTAIL (tail objects)

# state words of the attention core (the first ones are trip counts of its program). A_NOBJ: the A objects before the
# held ones; A_NKH: the sub-step group's K objects after the tails (a chain's first column, not split); A_NRM, A_VN, A_GS,
# A_QA (one per octet): the addresses of the norms, the step's V, the gates, the Q panels; A_HP: the held objects (in
# order), A_RW: the open records' rows (per tail object)
STATE = WordMap("attention core state", 50)
_A = STATE.common
(A_NNO, A_NXN, A_NXSKIP, A_NASKIP, A_NQ, A_NQP, A_NKV, A_NOBJ, A_NE, A_TP, A_TC, A_S, A_P, A_I, A_EIDX, A_ROLE, A_QS,
 A_KS, A_XL, A_QR, A_G0, A_GATT, A_GJ, A_NO, A_SA, A_NTAIL, A_SOFT, A_NKH, A_NRM, A_VN, A_GS) = _A.words(
    "A_NNO A_NXN A_NXSKIP A_NASKIP A_NQ A_NQP A_NKV A_NOBJ A_NE A_TP A_TC A_S A_P A_I A_EIDX A_ROLE A_QS A_KS A_XL A_QR "
    "A_G0 A_GATT A_GJ A_NO A_SA A_NTAIL A_SOFT A_NKH A_NRM A_VN A_GS", at=0)
A_KH = _A.word("A_KH")  # the KV head of the slots' K, V
A_QA = _A.block("A_QA", 4)  # up to 4 octets
A_HP = _A.block("A_HP", HOLD)
A_RW = _A.block("A_RW", TAILS)
A_LPER = _A.alias("A_LPER", A_TP)  # (a main core's phase of a packed design of 8 slots) log2 of its input objects' strips
A_WORDS = _A.next_free()
# (one pass) the new K's rows, panels; the merged records; a branch's K panels (its rows at their positions' lanes); a
# branching step's word (``M_BRANCH``: its first lane in its record | its slots << 8 | its record past the step's first
# << 12, or 0); from A_MK 4 words: the rows' masks of the step's first two records (``M_AMASK``: per record those of rows
# 0 .. 3, 4 .. 7, a row's lanes past its position); the slots' K lanes (``M_ALANE``: 3 bits a slot, a branch slot's at its
# position)
ONE = STATE.kind("one pass")
A_KR, A_KN, A_KM, A_KB, A_BR = ONE.words("A_KR A_KN A_KM A_KB A_BR", at=A_WORDS)
A_MK = ONE.block("A_MK", 4)
A_LN = ONE.word("A_LN")
# A_ROLE: a chain's first core (its K records), its last (V panels, rows), between
ROLE_K, ROLE_V, ROLE_O = _A.values("ROLE_K ROLE_V ROLE_O", of="role")


def consts() -> dict[str, int]:
    """The words and flags the attention core's kernels take (and the main core's, for the gates): the metadata words of its
    objects, its state words and the main core's gate words."""
    return XMETA.consts(PHASE, ATT, FLAGS) | STATE.consts(ONE) | GATES.consts()


_SRC = templating.load("models/qwen35/attn.cc.jinja")

_ONE = templating.load("models/qwen35/attn.one.cc.jinja")

# The main core's part of the gates (``AttnShape.main_gates``; decode.design's ``gate`` designs): the attention core's L1
# holds none. Rendered after the attention core's source (its helpers).
_MAIN_SRC = templating.load("models/qwen35/attn.main.cc.jinja")


# The main core's part of one pass: the gates of the query rows, per step (the attention core's L1 holds none)
_MAIN_ONE = templating.load("models/qwen35/attn.main_one.cc.jinja")


class Addr(NamedTuple):
    """Where a region of the core's state lies: ``base`` "ws" (its workspace), "mw" (the main core's workspace past its
    residual) or "mx" (the main core's sum panels), ``off`` bytes from the base."""

    base: Literal["ws", "mw", "mx"]
    off: int


class AttnShape:
    """The attention core's sizes for a model: QC query heads per column, NPH columns per KV head (a chain), the
    slots of a sub-step and their query-row sets, the cache's capacity, the workspace and its regions in the main core's
    memory."""

    def __init__(self, cfg: QwenConfig, max_positions: int, *, c_len: int = 64, a_bytes: int = OBJ_BYTES, xpay: int = 4096,
                 slots: int = SLOTS):
        """slots: a step's slots at most (a design of 8: ``one``)."""
        assert cfg.head_dim == HD and cfg.rope_dims == 2 * ROT and cfg.num_attention_heads % COLS == 0
        self.one = slots == 8  # (module docstring "One pass")
        self.smax = slots if self.one else SLOTS
        self.kvh = cfg.num_key_value_heads
        assert COLS % self.kvh == 0 and COLS // self.kvh >= 2 and 2 * HD * self.kvh <= xpay
        self.nph = COLS // self.kvh
        self.qc = cfg.num_attention_heads // COLS
        assert 1 <= self.qc <= 4 and 2 * 2 * self.qc * HD <= xpay and 4 * (2 * HD + 64 * self.smax) <= xpay
        assert REC.pg(a_bytes) == POS_PER_GROUP
        # (a sub-step's slots: as many as the core holds the chain's query rows for; then a step takes up to u
        # sub-steps, one more where its slots cross a group. A dense model's main core has the program memory to apply
        # the gates: then the core takes more rows)
        self.sub1 = slots if self.one else min(SLOTS, (MAX_ROWS if cfg.moe else MAX_ROWS_G) // (self.nph * self.qc))
        assert self.sub1 >= 1
        # (more slots than a chain of 4 holds: two rounds, one per KV head, of both chains, each with half its heads; the
        # first chain's shims read the round's cache for the column 4 away too)
        self.rounds = not self.one and self.nph == 4 and self.kvh == 2 and self.sub1 < SLOTS
        self.a_bcast = tuple((c, c + 4) for c in range(COLS // 2)) if self.rounds else ()
        self.sub = SLOTS if self.rounds else self.sub1
        # (a step's sub-steps at most, a round each: one more where a sub-step of more slots than sub1 runs in two)
        self.u = 1 if self.one else -(-(slots - 1) // self.sub) + 1 + self.rounds
        self.rows = 8 * self.qc if self.one else self.sub1 * self.nph * self.qc  # (a sub-step's query rows at most)
        self.main_gates = self.one or self.rows > MAX_ROWS
        assert not (self.rounds and self.main_gates)  # (``_MAIN_SRC``: a chain's last column gates o)
        self.octets = -(-self.rows // 8)
        self.sets = 2 * self.octets
        self.po = PO
        self.ps = rup(self.po * self.octets, 64)  # (a group's P panels: a panel stream starts 64-byte aligned)
        assert A_QA + self.octets <= A_HP and self.octets <= 3  # (pvs)
        self.c_len, self.a_bytes = c_len, a_bytes
        self.r_bytes = 2 * c_len * (2 if self.main_gates else 1)  # (an R object: o fp32 when the main core gates it)
        self.eps = cfg.rms_norm_eps
        self.groups = -(-max_positions // POS_PER_GROUP)
        self.max_positions = self.groups * POS_PER_GROUP
        self.head_bytes = OBJS * a_bytes * self.groups  # a KV head's cache
        self._plan()

    def _plan(self) -> None:
        """The workspace (the core's L1 next to its two A buffers, X buffer, R buffers, token and stack) and the regions
        in the main core's memory (``ext``: an ``Addr`` of "mw" its workspace past its residual or "mx" its sum panels):
        the gates there (unless the main core applies them), the norms and the step's V too where the octets' Q panels
        then fit the core's L1, else the Q panels that do not."""
        y = ("y", 4 * 32 * (ND + 1) * self.sets)
        own = [("st", 4 * (STATE.size if self.one else A_WORDS)), y, ("ml", 4 * 64 * self.sets), ("sc", 4 * 128 * self.sets),
               ("pp", self.ps)]
        self.gate_tab = not self.one and not self.main_gates  # (the sigmoid's table in the core's L1)
        fixed = 2 * self.a_bytes + XB + 2 * self.r_bytes + 64 + STACK + (SIGMOID.table_words().nbytes if self.gate_tab else 0)
        self.ext_w, self.ext_x = (EXT_W8, EXT_X8) if self.one else (EXT_W, EXT_X)
        free = {"mw": self.ext_w, "mx": self.ext_x}
        self.ext: dict[str, Addr] = {}

        def put(name: str, n: int, order: tuple[str, ...] = ("mw", "mx")) -> None:
            r = next(r for r in order if free[r] >= n)
            self.ext[name] = Addr(r, (self.ext_w if r == "mw" else self.ext_x) - free[r])
            free[r] -= n

        cold = [("nrm", 4 * (2 * HD + 64 * self.smax)), ("vn", 2 * HD * self.smax)]
        if self.one:
            for name, n in cold:
                put(name, n)
            put("kr", ROWS_B, ("mx", "mw"))
            put("km", 2 * REC_K, ("mx", "mw"))
            put("kb", REC_K, ("mx", "mw"))
            pan = PANEL_BYTES * ND
            own += [("qa0", min(self.qc, 2) * pan), ("qa1", pan)] if self.qc > 2 else [("qa0", self.qc * pan)]
            own.append(("kn", REC_K))
            assert ROWS_B + 2 * REC_V + PANEL_BYTES * self.qc <= self.a_bytes  # (tobj: past a tail's rows its V pair, the P panels)
        else:
            own.append(("kb", max(9 * ND * self.smax, REC_V)))  # (the slots' K blocks; aemit: a record's V panels)
            qa = [(f"qa{o}", 2 * PANEL_BYTES * ND) for o in range(self.octets)]
            if not self.main_gates:
                put("gs", 2 * self.rows * HD)
                put(*cold[1], ("mx",))
                own += cold[:1] + qa
            else:
                for name, n in cold:
                    put(name, n)
                n_own = (L1_BYTES - fixed - regions(own)["size"]) // qa[0][1]
                own += qa[:n_own]
                for name, n in qa[n_own:]:
                    put(name, n, ("mx", "mw"))
            assert "qa0" in dict(own) and self.octets >= 2 and 2 * REC_V <= 2 * PANEL_BYTES * ND  # (tobj: a tail's V pair in the
            # second octet's Q panels)
        self.mx_bytes = self.ext_x - free["mx"]  # (of the main core's sum panels)
        self._ws = regions(own)
        assert self._ws["size"] + fixed <= L1_BYTES, (self._ws["size"], fixed)
        assert 4 * 128 * self.sets >= 4 * HD  # (kf: the scores' region)

    def at(self, tail: int) -> list[tuple[int, Addr]]:
        """The address state words abegin sets: (word, address); tail: the main core's workspace offset past its
        residual."""
        out = []
        names = [(A_NRM, "nrm"), (A_VN, "vn")] + [(A_GS, "gs")] * (not self.main_gates)
        names += [(A_KR, "kr"), (A_KN, "kn"), (A_KM, "km"), (A_KB, "kb")] if self.one else []
        for w, name in names + [(A_QA + o, f"qa{o}") for o in range((1 + (self.qc > 2)) if self.one else self.octets)]:
            if name in self.ext:
                a = self.ext[name]
                out.append((w, Addr(a.base, a.off + (tail if a.base == "mw" else 0))))
            else:
                out.append((w, Addr("ws", self._ws[name])))
        return out

    def role(self, c: int) -> int:
        """Column c's role in its KV head's chain (``ROLE_K``: its first, ``ROLE_V``: its last, ``ROLE_O``)."""
        k = c % self.nph
        return ROLE_K if k == 0 else ROLE_V if k == self.nph - 1 else ROLE_O

    def n_rel(self, c: int, split: int, n: int) -> int:
        """R objects of a sub-step of n slots of column c: o (split: the chain's last column's of the chain's heads, a
        round's: of its half of them), then two records (the chain's first column K records, its last one V panels and
        rows; a round: the KV head's own chain)."""
        role = ROLE_O if split & SPLIT_NOREC else self.role(c)
        last = c % self.nph == self.nph - 1
        if split & 3 == SPLIT_ROUND:
            no = 4 * n * 2 * self.qc * last
        else:
            no = 32 * self.qc if self.one else 4 * n * ((self.nph * self.qc if last else 0) if split else self.qc)
        nr = self.records(n)
        k = REC_K if self.one else 2 * REC_K  # (one pass: a K record; sub-steps: its pair)
        return no + {ROLE_K: nr * k, ROLE_V: nr * (REC_V + ROWS_B), ROLE_O: 0}[role] // (2 * self.c_len)

    @staticmethod
    def splits(g: int) -> bool:
        """Whether a sub-step whose slots are in group g splits the cache along the chains (from 4 groups on: the
        same for each of its positions in every step)."""
        return g >= 3

    def records(self, n: int) -> int:
        """A sub-step of n slots' records out and tail objects: of its first and last positions (one pass: the same
        record twice when one holds both; sub-steps of one slot: one)."""
        return 2 if self.one or n > 1 else 1

    def split_of(self, g: int, n: int) -> tuple[int, ...]:
        """The splits of a sub-step of n slots in group g, a round each: 0, SPLIT_CHAIN, or more slots than a chain
        holds: ``SPLIT_ROUND`` of KV head 0, then 1 (M_SPLIT's bits less the no-records one: ``round_word``)."""
        if not self.splits(g):
            return (0,)
        return (SPLIT_ROUND, SPLIT_ROUND | SPLIT_KV) if n > self.sub1 else (SPLIT_CHAIN,)

    def round_word(self, split: int, c: int) -> int:
        """Column c's M_SPLIT of a sub-step's split: a round's no-records bit where c's chain is not the KV head's."""
        return split | (SPLIT_NOREC if split & 3 == SPLIT_ROUND and c // self.nph != (split & SPLIT_KV) >> 2 else 0)

    def bounds(self, g: int) -> list[int]:
        """A split sub-step in group g: the groups of a chain's cores, core k's from bounds[k] to bounds[k + 1]."""
        G, n = g + 1, self.nph
        assert G >= n
        return [G * k // n for k in range(n + 1)]

    def workspace(self) -> dict[str, int]:
        """Byte offsets of the core's workspace regions (and "size"). y: per set its blocks [db][row 4 x 8] fp32, then
        l (before the panels: the q rows [row][256] fp32); sc: per set the group's scores [record 4][row 4 x 8] (before
        the cache: a slot's K, fp32); pp: a group's P panels (``ps`` bytes) [octet][P0 | P1 | P2 | P3]; qa<o>: the Q panels, [d-block][octet 2]
        (qa1: octets 1, 2 of three; done with in the tails: qa1 a tail's V pair)."""
        return dict(self._ws)

    def render(self, p: str, *, tail: int, nx_per_norm: int, packed8: bool = False) -> str:
        """nx_per_norm: the design's input objects per norm-weight object; packed8: the main core's inputs packed in a design of
        8 slots (``decode.protocol.M_SP``)."""
        v = dict(p=p, a=Attr(self.workspace()), at=self.at(tail), QC=self.qc, NPH=self.nph, SETS=self.sets, NO=self.octets, SMAX=self.smax,
                 HD=HD, ND=ND, HOLD=HOLD, PS=self.ps, PO=self.po, REC_K=REC_K, REC_V=REC_V, ROWS_B=ROWS_B, eps=repr(self.eps),
                 C_LEN=self.c_len, META=META, NX_PER_NORM=nx_per_norm, OBJ=self.a_bytes, PAIR_K=2 * REC_K, NO_OBJ=32 * self.qc,
                 GATE_MAIN=self.main_gates, PACKED8=packed8, ROUNDS=self.rounds,
                 SPLIT_ROUND=SPLIT_ROUND, SPLIT_NOREC=SPLIT_NOREC,
                 **consts())
        if self.one:
            return _ONE.render(**v) + _MAIN_ONE.render(**v)
        return _SRC.render(**v) + (_MAIN_SRC.render(**v) if self.main_gates else "")


def step_words(p: int, S: int, branch: tuple[int, int] | None = None) -> tuple[int, np.ndarray, int]:
    """(one pass) A step's F_ATT words that follow its slots' positions (p, S slots; ``branch``: its last L slots follow slot c,
    at p + c + 1 .., inside the 8 positions of c's record): the branch word (``M_BRANCH``: its first lane in its record | L << 8
    | its record past the step's first << 12, or 0), the rows' masks of the step's first two records (``M_AMASK``: per
    record rows 0 .. 3, 4 .. 7, a row's lanes past its position: slot s's p + s, a branch slot's its own), the slots' K lanes
    (``M_ALANE``: their positions' lanes, 3 bits each)."""
    c, L = branch or (-1, 0)
    sc = S - L
    q = p % 8 + c + 1  # (the branch's first position, from the step's first record's)
    word = q % 8 | L << 8 | q // 8 << 12 if branch else 0
    pos = [p % 8 + (s if s < sc else c + 1 + s - sc) for s in range(8)]
    w = np.zeros(4, np.uint32)
    for k in range(2):
        for s in range(8):
            n = min(max(pos[s] - 8 * k + 1, 0), 8)
            w[2 * k + s // 4] |= ((0xFF << n) & 0xFF) << (8 * (s % 4))
    return word, w, sum((pos[s] % 8) << (3 * s) for s in range(8))


# ----------------------------------------------------------------------
# Host side: the RoPE angles
# ----------------------------------------------------------------------


def rope_cs(cfg: QwenConfig, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cos, sin (n, 32) fp32 of positions' angles (``QwenConfig.rope_theta``: (n,) text positions or (n, 3) (T, H, W))."""
    rows = cos_sin_rows(cfg.rope_theta(positions))
    return np.ascontiguousarray(rows[:, :ROT]), np.ascontiguousarray(rows[:, ROT:])
