"""Minimal parser of AIE full ELFs (``aiecc --generate-full-elf``) and the patching of their main control program.

Handles groups of one ``.ctrltext`` each, uncompressed sections, relocation types 5 and 8 (``patch_control_code``
rejects anything else rather than load control code it did not patch).
"""

import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

# ELF32 entry sizes
_SYM_SIZE = 16
_RELA_SIZE = 12

# Relocation symbol types (from XRT elf_patcher.h)
PATCH_SHIM_DMA_48 = 5
PATCH_ADDRESS_64 = 8

SHF_COMPRESSED = 0x800

CTRL_SCRATCHPAD = "scratch-pad-ctrl"  # the symbol of the control program's scratchpad (runtime parameters)


@dataclass
class ElfSection:
    name: str
    index: int
    offset: int
    size: int
    addr: int
    data: bytes
    flags: int = 0


@dataclass
class ElfSymbol:
    name: str
    value: int
    size: int
    section_index: int


@dataclass
class ElfRelocation:
    offset: int
    sym_index: int
    rela_type: int
    addend: int


@dataclass
class ElfGroup:
    """A group of sections: typically just a .ctrltext.N reference."""

    group_id: int
    ctrltext: ElfSection | None = None


@dataclass
class ParsedElf:
    """Parsed ELF with groups, symbols, and relocations."""

    groups: dict[int, ElfGroup] = field(default_factory=dict)
    symbols: list[ElfSymbol] = field(default_factory=list)
    relocations: list[ElfRelocation] = field(default_factory=list)
    section_to_group: dict[int, int] = field(default_factory=dict)
    sections: list[ElfSection] = field(default_factory=list)
    pdi_map: dict[str, ElfSection] = field(default_factory=dict)
    partition_columns: int | None = None  # first word of .note.xrt.configuration


def _read_u16(data: bytes, off: int) -> int:
    return struct.unpack_from("<H", data, off)[0]


def _read_u32(data: bytes, off: int) -> int:
    return struct.unpack_from("<I", data, off)[0]


def _read_string(strtab: bytes, offset: int) -> str:
    end = strtab.index(b"\x00", offset)
    return strtab[offset:end].decode("ascii")


def parse_elf(path: str | Path) -> ParsedElf:
    """Parse an AIE full-ELF file."""
    data = Path(path).read_bytes()

    # ELF header
    e_shoff = _read_u32(data, 32)
    e_shentsize = _read_u16(data, 46)
    e_shnum = _read_u16(data, 48)
    e_shstrndx = _read_u16(data, 50)

    # Parse section headers
    sections: list[ElfSection] = []
    for i in range(e_shnum):
        base = e_shoff + i * e_shentsize
        sh_name_off = _read_u32(data, base)  # noqa: F841
        sh_flags = _read_u32(data, base + 8)
        sh_addr = _read_u32(data, base + 12)
        sh_offset = _read_u32(data, base + 16)
        sh_size = _read_u32(data, base + 20)
        sec_data = data[sh_offset : sh_offset + sh_size] if sh_size > 0 else b""
        sections.append(
            ElfSection(
                name="",  # filled below
                index=i,
                offset=sh_offset,
                size=sh_size,
                addr=sh_addr,
                data=sec_data,
                flags=sh_flags,
            )
        )

    # Resolve section names from shstrtab
    shstrtab = sections[e_shstrndx].data
    for sec in sections:
        base = e_shoff + sec.index * e_shentsize
        name_off = _read_u32(data, base)
        sec.name = _read_string(shstrtab, name_off)

    compressed = [s.name for s in sections if s.flags & SHF_COMPRESSED]
    if compressed:
        raise ValueError(f"{path}: compressed sections are not supported: {compressed}")

    # Collect PDI sections globally (deduplicated by name, first occurrence wins)
    pdi_map: dict[str, ElfSection] = {}
    for sec in sections:
        if sec.name.startswith(".pdi.") and sec.name not in pdi_map:
            pdi_map[sec.name] = sec

    # Build section-to-group mapping from .group.N sections
    section_to_group: dict[int, int] = {}
    groups: dict[int, ElfGroup] = {}

    for sec in sections:
        if sec.name.startswith(".group."):
            grp_id = int(sec.name.split(".")[-1])
            groups[grp_id] = ElfGroup(group_id=grp_id)
            # Group section data: first word is flags (GRP_COMDAT=1), rest are member indices
            n_entries = sec.size // 4
            for j in range(1, n_entries):  # skip flags word
                member_idx = _read_u32(sec.data, j * 4)
                if member_idx < len(sections):
                    section_to_group[member_idx] = grp_id

    # Assign ctrltext sections to groups
    for sec_idx, grp_id in section_to_group.items():
        sec = sections[sec_idx]
        if sec.name.startswith(".ctrltext."):
            if groups[grp_id].ctrltext is not None:
                raise ValueError(f"{path}: group {grp_id} has several .ctrltext sections")
            groups[grp_id].ctrltext = sec

    # Parse .dynsym + .dynstr
    symbols: list[ElfSymbol] = []
    dynsym_sec = next((s for s in sections if s.name == ".dynsym"), None)
    dynstr_sec = next((s for s in sections if s.name == ".dynstr"), None)
    if dynsym_sec and dynstr_sec:
        n_syms = dynsym_sec.size // _SYM_SIZE
        for i in range(n_syms):
            base = i * _SYM_SIZE
            st_name = _read_u32(dynsym_sec.data, base)
            st_value = _read_u32(dynsym_sec.data, base + 4)
            st_size = _read_u32(dynsym_sec.data, base + 8)
            st_shndx = _read_u16(dynsym_sec.data, base + 14)
            name = _read_string(dynstr_sec.data, st_name)
            symbols.append(
                ElfSymbol(
                    name=name, value=st_value, size=st_size, section_index=st_shndx
                )
            )

    # Parse .rela.dyn
    relocations: list[ElfRelocation] = []
    rela_sec = next((s for s in sections if s.name == ".rela.dyn"), None)
    if rela_sec:
        n_rela = rela_sec.size // _RELA_SIZE
        for i in range(n_rela):
            base = i * _RELA_SIZE
            r_offset = _read_u32(rela_sec.data, base)
            r_info = _read_u32(rela_sec.data, base + 4)
            r_addend = _read_u32(rela_sec.data, base + 8)
            sym_idx = r_info >> 8
            rela_type = r_info & 0xFF
            relocations.append(
                ElfRelocation(
                    offset=r_offset,
                    sym_index=sym_idx,
                    rela_type=rela_type,
                    addend=r_addend,
                )
            )

    # .note.xrt.configuration: namesz, descsz, type, name (padded to 4 bytes), desc
    partition_columns = None
    note = next((s for s in sections if s.name == ".note.xrt.configuration"), None)
    if note is not None and note.size >= 12:
        namesz, descsz = _read_u32(note.data, 0), _read_u32(note.data, 4)
        desc = 12 + (namesz + 3) // 4 * 4
        if descsz >= 4 and desc + 4 <= note.size:
            partition_columns = _read_u32(note.data, desc)

    return ParsedElf(
        groups=groups,
        symbols=symbols,
        relocations=relocations,
        section_to_group=section_to_group,
        sections=sections,
        pdi_map=pdi_map,
        partition_columns=partition_columns,
    )


class DataBd(NamedTuple):
    """A shim DMA BD of the control program that addresses a runtime-sequence argument."""

    offset: int  # byte offset of the BD in the control program
    arg: str  # the argument (its symbol: its index in the runtime sequence)
    start: int  # byte offset in the argument where the transfer starts
    words: int  # the transfer's length in 32-bit words


def main_group(elf: ParsedElf) -> ElfGroup:
    """The group of the main control program (the largest .ctrltext)."""
    group = max(elf.groups.values(), key=lambda g: g.ctrltext.size if g.ctrltext else 0)
    assert group.ctrltext is not None
    return group


def pdi_sections(elf: ParsedElf) -> dict[str, ElfSection]:
    """The PDI sections the main control program references, by symbol name."""
    text = main_group(elf).ctrltext.index
    out: dict[str, ElfSection] = {}
    for rela in elf.relocations:
        sym = elf.symbols[rela.sym_index]
        if (sym.name.startswith(".pdi.") and rela.rela_type == PATCH_ADDRESS_64 and sym.section_index == text
                and sym.name in elf.pdi_map):
            out.setdefault(sym.name, elf.pdi_map[sym.name])
    return out


def patch_control_code(elf: ParsedElf, *, address: Callable[[DataBd], int], pdis: Mapping[str, int],
                       scratchpad: int | None) -> tuple[bytearray, list[DataBd]]:
    """The main control program with every relocation resolved, and its data BDs: each one's transfer at the device
    address ``address(bd)``, the PDIs at ``pdis`` (by symbol), the scratchpad at ``scratchpad``."""
    group = main_group(elf)
    text = group.ctrltext.index
    members = [elf.sections[i].name for i, g in elf.section_to_group.items() if g == group.group_id and i != text]
    if members:  # (.ctrldata, control packets, preemption save / restore: never patched here)
        raise ValueError(f"main ELF group has sections besides its .ctrltext: {members}")
    data = bytearray(group.ctrltext.data)
    bds: list[DataBd] = []
    for rela in elf.relocations:
        sym = elf.symbols[rela.sym_index]
        if sym.section_index != text:
            continue
        if rela.rela_type == PATCH_SHIM_DMA_48:  # (BD words 1, 2: address bits 31:2, 47:32 in the low half)
            words, lo, hi = struct.unpack_from("<III", data, rela.offset)
            bd = DataBd(rela.offset, sym.name, ((hi & 0xFFFF) << 32 | lo) + rela.addend, words)
            bds.append(bd)
            addr = address(bd)
            struct.pack_into("<II", data, rela.offset + 4, addr & 0xFFFFFFFC, (hi & 0xFFFF0000) | (addr >> 32 & 0xFFFF))
        elif rela.rela_type == PATCH_ADDRESS_64 and sym.name in pdis:
            struct.pack_into("<Q", data, rela.offset, pdis[sym.name])
        elif rela.rela_type == PATCH_ADDRESS_64 and sym.name == CTRL_SCRATCHPAD and scratchpad is not None:
            struct.pack_into("<Q", data, rela.offset, scratchpad)
        else:
            raise ValueError(f"unsupported relocation of the control code: type {rela.rela_type}, symbol "
                             f"{sym.name!r} at {rela.offset:#x}")
    return data, bds
