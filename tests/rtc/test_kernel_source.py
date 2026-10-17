"""Kernel sources named by their content (``operators/kernel_source.py``): host only."""

import re

import pytest

from artlite.operators.kernel_source import KernelSource, KernelSourceError, kernel_prefix, split, translation_unit

P = kernel_prefix("op")
L = kernel_prefix("lut")


def _helper(scale: int) -> str:
    return f"""
// the helper
[[clang::always_inline]] inline int {P}_h(int x) {{ return x * {scale}; }}
"""


def _source(scale: int, n: int) -> str:
    return f"""
#define AW(T, off) ((T *)((uint8_t *)ws + (off)))

namespace {{
{_helper(scale)}
alignas(64) static const uint32_t {L}_ab[4] = {{1, 2, 3, 4}};
alignas(64) static const uint32_t {L}_cd[4] = {{1, 2, 3, 4}};
}}  // namespace

extern "C" {{

void {P}_entry(int32_t *ws) {{  // the entry
    for (int i = 0; i < {n}; i++)
        AW(int32_t, 4)[i] = {P}_h(i) + (int)({L}_ab[i % 4] + {L}_cd[i % 4]);
}}

}}  // extern "C"
"""


def _defs(text: str) -> set[str]:
    return set(re.findall(r"\b(op_\w+_[0-9a-f]{8}|lut_\w+_[0-9a-f]{8})\b", text))


def test_split() -> None:
    items = split(_source(2, 8))
    assert [i.defines for i in items] == [("AW",), (f"{P}_h",), (f"{L}_ab",), (f"{L}_cd",), (f"{P}_entry",)]
    assert [i.wrapper for i in items] == ["", "namespace", "namespace", "namespace", "C"]
    assert items[1].text.lstrip().startswith("// the helper") and "// the entry" in items[4].text
    assert items[4].refs == {f"{P}_h", f"{L}_ab", f"{L}_cd"} and KernelSource(_source(2, 8)).items[4].macro_refs == ("AW",)


def test_names_follow_the_code() -> None:
    a, b, c = (KernelSource(_source(s, n), label=f"s{s} n{n}") for s, n in ((2, 8), (2, 16), (3, 8)))
    # the helper and tables: equal in a and b; the entries differ (trip count)
    assert a.name(f"{P}_h") == b.name(f"{P}_h") != c.name(f"{P}_h")
    assert a.name(f"{P}_entry") != b.name(f"{P}_entry")
    # a different helper renames everything that reaches it
    assert a.name(f"{L}_ab") == c.name(f"{L}_ab") and a.name(f"{P}_entry") != c.name(f"{P}_entry")
    # identical tables under different names stay two tables
    assert a.name(f"{L}_ab") != a.name(f"{L}_cd")
    assert re.fullmatch(r"op_h_[0-9a-f]{8}", a.name(f"{P}_h"))


def test_translation_unit_holds_each_definition_once() -> None:
    a, b, c = (KernelSource(_source(s, n), label=f"s{s} n{n}") for s, n in ((2, 8), (2, 16), (3, 8)))
    tu = translation_unit([a, b, c])
    for ks in (a, b, c):
        for item, text in zip(ks.items, ks.texts, strict=True):
            if item.macro is None:
                assert tu.count(text.strip()) == 1, text
    assert len(_defs(tu)) == 7  # (h twice, the tables once, three entries)
    assert tu.count("#define AW") == 1 and tu.count("// ==== ") == 3
    assert f"shared with an earlier section: {a.name(f'{P}_h')}" in tu  # (b's header)
    assert "__artlite_" not in tu


def test_macro_context() -> None:
    """An unprefixed macro is part of the code that uses it: another definition renames its users and is emitted
    before them."""
    a = KernelSource(_source(2, 8))
    b = KernelSource(_source(2, 8).replace("(uint8_t *)ws + (off)", "(uint8_t *)ws + 2 * (off)"))
    assert a.name(f"{P}_h") == b.name(f"{P}_h") and a.name(f"{P}_entry") != b.name(f"{P}_entry")
    tu = translation_unit([a, b])
    assert tu.count("#define AW") == 2
    assert tu.index("2 * (off)") < tu.index(b.name(f"{P}_entry")) and tu.index("+ (off)") < tu.index("2 * (off)")


def test_declaration_and_definition() -> None:
    src = f"""
extern "C" {{
void {P}_a(int x);
}}
namespace {{
void {P}_b(int x) {{ if (x) {P}_a(x - 1); }}
}}
extern "C" {{
void {P}_a(int x) {{ {P}_b(x); }}
void {P}_c(void) __attribute__((alias("{P}_a")));
}}
"""
    ks = KernelSource(src)
    names = ks.names
    assert len({names[f"{P}_a"][-8:], names[f"{P}_b"][-8:]}) == 1  # (one cycle: one hash)
    assert f'alias("{names[f"{P}_a"]}")' in ks.text


def test_rename_and_errors() -> None:
    ks = KernelSource(_source(2, 8))
    mlir = f'func.func private @{P}_entry(memref<64xi32>) attributes {{link_with = "kernel.o"}}'
    assert ks.rename(mlir) == mlir.replace(f"{P}_entry", ks.name(f"{P}_entry"))
    with pytest.raises(KernelSourceError, match="not defined by the kernel source"):
        ks.rename(f"@{P}_missing")
    with pytest.raises(KernelSourceError, match="used but not defined"):
        KernelSource(f"void {P}_x() {{ {P}_y(); }}")


def test_fixed_names() -> None:
    """A source without placeholder names (fixed kernel names) is emitted once per text."""
    src = 'extern "C" {\nvoid add_bf16(int *a) { a[0] = 1; }\n}  // extern "C"\n'
    tu = translation_unit([KernelSource(src), KernelSource(src)])
    assert tu.count("void add_bf16") == 1
