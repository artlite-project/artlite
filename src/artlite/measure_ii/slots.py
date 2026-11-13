"""Issue slots and shared ports of AIE2P assembly mnemonics, and the per-loop op counts."""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from .. import aie2p_isa

# The issue slot and the shared ports of each AIE2P assembly mnemonic: aie2p_isa, generated from llvm-aie's TableGen
# sources (python -m artlite.gen_isa_tables after a toolchain upgrade). Mnemonics of several slots (mov, mov.nb, vmov,
# j, jl) are told apart by their operands in slot_of; nops are not counted. A lng (long) op takes the alu and the mv
# slot of its bundle.
#
# The ports: a loop's II is at least the number of its ops on each, which the per-slot counts do not show (e.g.
# vshuffle and the conversion pushes / flushes of a bfp16 output stream all read L_RM_PORT: a dequant loop of 30
# shuffles and 8 panel pushes runs at II 46 with its mv slot at 30).
SLOTS = tuple(aie2p_isa.SLOT_MNEMONICS)  # lda ldb st vec mv alu lng
_SLOT_OF = {mnemonic: slot for slot, text in aie2p_isa.SLOT_MNEMONICS.items() for mnemonic in text.split()}
PORTS = tuple(aie2p_isa.PORT_UNITS)
_PORTS_OF: dict[str, list[tuple[str, int]]] = {}  # mnemonic -> [(port, cycles)]
for _port, _text in aie2p_isa.PORT_MNEMONICS.items():
    for _word in _text.split():
        _mnemonic, _, _cycles = _word.partition("*")
        _PORTS_OF.setdefault(_mnemonic, []).append((_port, int(_cycles or 1)))
# (mnemonics whose ports follow their first operand's register kind: vst of a vector register reads QEY_RS, of an
# accumulator not)
_PORT_KINDS = {mnemonic: {kind: list(busy.items()) for kind, busy in by.items()}
               for mnemonic, by in aie2p_isa.PORT_KINDS.items()}
_REGISTER_KIND = re.compile(r"(bm|cm|dm|ex|ey|wl|wh|x|y)[lh]{0,2}\d")
_KIND = {"bm": "BM", "cm": "CM", "dm": "DM", "ex": "EX", "ey": "EY", "wl": "W", "wh": "W", "x": "X", "y": "Y"}


def ports_of(instruction: str) -> list[tuple[str, int]]:
    """The shared ports (and cycles) of an assembly instruction."""
    mnemonic, *rest = instruction.split(None, 1)
    by = _PORT_KINDS.get(mnemonic)
    if by is not None and rest:
        found = _REGISTER_KIND.match(rest[0].strip())
        if found and _KIND[found.group(1)] in by:
            return by[_KIND[found.group(1)]]
    return _PORTS_OF.get(mnemonic, [])


def slot_of(instruction: str) -> str | None:
    """The issue slot of an assembly instruction: one of SLOTS, None for a nop, "?" for an unknown mnemonic."""
    mnemonic, *rest = instruction.split(None, 1)
    operands = rest[0].strip() if rest else ""
    if mnemonic.startswith("nop"):
        return None
    if mnemonic in ("mov", "mov.nb"):
        if ".flt2fx" in operands or ".fx2flt" in operands:
            return "alu"
        if re.match(r"ms\b", operands):
            return "st"
        if mnemonic == "mov.nb" or re.search(r",\s*ss$", operands):
            return "lda"
        return "mv"
    if mnemonic == "vmov":
        if re.match(r"mcd\b", operands):
            return "st"
        return "lda" if re.search(r"\bscd\b", operands) else "mv"
    if mnemonic in ("j", "jl"):
        return "lng" if "#" in operands else "alu"
    return _SLOT_OF.get(mnemonic, "?")


@dataclass
class SlotCounts:
    """The ops of a loop body (one iteration) per issue slot, and per shared port (PORTS)."""

    ops: dict[str, Counter[str]]  # slot -> mnemonic -> count ("?": unknown mnemonics)
    bundles: int = 0
    port_ops: dict[str, Counter[str]] = field(default_factory=dict)  # port -> mnemonic -> cycles

    def count(self, slot: str) -> int:
        if slot in PORTS:
            return sum(self.port_ops.get(slot, Counter()).values())
        return sum(self.ops.get(slot, Counter()).values())

    @property
    def pressure(self) -> dict[str, int]:
        """Bundles each slot and port needs per iteration: a lng op occupies the alu and the mv slot."""
        lng = self.count("lng")
        slots = {slot: self.count(slot) + (lng if slot in ("alu", "mv") else 0) for slot in SLOTS if slot != "lng"}
        return slots | {port: self.count(port) for port in PORTS}

    @property
    def bound(self) -> int:
        """The resource bound of the II: the busiest slot or port."""
        return max(self.pressure.values(), default=0)

    @property
    def binding(self) -> list[str]:
        bound = self.bound
        return [slot for slot, count in self.pressure.items() if bound and count == bound]

    @property
    def busiest_port(self) -> tuple[str, int] | None:
        counts = [(self.count(port), port) for port in PORTS if self.count(port)]
        return (max(counts)[1], max(counts)[0]) if counts else None

    def describe(self, slot: str) -> str:
        """The slot's (or port's) ops, most frequent first: 'vshuffle 16, vband 6'."""
        ops = self.port_ops.get(slot, Counter()) if slot in PORTS else self.ops.get(slot, Counter())
        if slot in ("alu", "mv"):
            ops = ops + self.ops.get("lng", Counter())
        return ", ".join(f"{mnemonic} {count}" for mnemonic, count in ops.most_common())


def count_slots(bundles: Sequence[Sequence[str]]) -> SlotCounts:
    """Per-slot and per-port op counts of assembly bundles (each a list of instructions)."""
    counts = SlotCounts(ops={}, bundles=len(bundles))
    for bundle in bundles:
        for instruction in bundle:
            slot = slot_of(instruction)
            if slot is not None:
                mnemonic = instruction.split(None, 1)[0]
                counts.ops.setdefault(slot, Counter())[mnemonic] += 1
                for port, cycles in ports_of(instruction):
                    counts.port_ops.setdefault(port, Counter())[mnemonic] += cycles
    return counts
