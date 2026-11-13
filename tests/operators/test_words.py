"""``operators.words``: declared word layouts check themselves (host only)."""

import pytest

from artlite.operators.words import LayoutError, WordMap


def test_next_free_and_blocks() -> None:
    m = WordMap("m", 16)
    h = m.common
    a, b = h.words("A B")
    blk = h.block("BLK", 4)
    c = h.word("C")
    assert (a, b, blk, c) == (0, 1, 2, 6)
    assert h.consts() == {"A": 0, "B": 1, "BLK": 2, "C": 6}
    assert h.spans()["BLK"] == (2, 4)


def test_bounds_and_overlap_in_a_kind() -> None:
    m = WordMap("m", 8)
    h = m.common
    h.words("A B C", at=0)
    with pytest.raises(LayoutError, match="outside"):
        h.word("X", at=8)
    with pytest.raises(LayoutError, match="outside"):
        h.block("X", 3, at=6)
    with pytest.raises(LayoutError, match="overlaps"):
        h.block("X", 2, at=2)
    with pytest.raises(LayoutError, match="twice"):
        h.word("A", at=5)
    with pytest.raises(LayoutError, match="outside"):
        m.kind("k").word("Y", at=-1)


def test_next_free_skips_what_a_kind_meets() -> None:
    m = WordMap("m", 16)
    m.common.word("TAIL", at=6)
    k = m.kind("k")
    k.word("X", at=5)
    assert k.word("Y") == 7  # (the common word at 6 is skipped)
    assert k.block("Z", 4) == 8
    with pytest.raises(LayoutError, match="outside"):
        k.block("W", 9)


def test_common_words_are_every_kinds() -> None:
    m = WordMap("m", 16, objects=True)
    m.common.words("H0 H1 H2", at=0)
    k = m.kind("k")
    assert k.word("X", at=3) == 3
    with pytest.raises(LayoutError, match="overlaps common"):
        k.word("Y", at=2)


def test_objects_overlap_unless_extended() -> None:
    m = WordMap("m", 16, objects=True)
    m.common.words("H0 H1", at=0)
    a, b = m.kind("a"), m.kind("b")
    a.word("X", at=5)
    b.word("Y", at=5)  # (different objects: the same word)
    c = m.kind("c", extends=(a,))
    with pytest.raises(LayoutError, match="overlaps a's X"):
        c.word("Z", at=5)
    assert c.word("Z") == 2  # (the first free word: past the common ones)
    # (the extended kind sees the words of the kinds that extend it)
    with pytest.raises(LayoutError, match="overlaps c's Z"):
        a.word("W", at=2)
    d = m.kind("d", extends=(c,))  # (two levels)
    with pytest.raises(LayoutError, match="overlaps a's X"):
        d.word("V", at=5)
    assert m.consts(d) == {"H0": 0, "H1": 1, "X": 5, "Z": 2}


def test_features_overlap_unless_exclusive() -> None:
    m = WordMap("m", 16)
    post = m.kind("post")
    tok = m.kind("tok")
    moe = m.kind("moe", exclusive_with=(post, tok))
    post.word("S_POST", at=4)
    tok.word("S_TOK", at=8)
    with pytest.raises(LayoutError, match="overlaps post"):
        tok.word("S_X", at=4)
    moe.words("S_A S_B", at=4)  # (never with post, tok)
    moe.word("S_C", at=8)
    m.together(post, tok)
    with pytest.raises(LayoutError, match="never meet"):
        m.together(post, moe)
    with pytest.raises(LayoutError, match="extend one another"):
        m.exclusive(m.kind("e", extends=(post,)), post)


def test_kinds_of_two_maps_do_not_mix() -> None:
    a, b = WordMap("a", 4), WordMap("b", 4)
    with pytest.raises(LayoutError, match="another map"):
        a.kind("k", extends=(b.kind("j"),))


def test_flags_values_and_aliases() -> None:
    m = WordMap("m", 16, objects=True)
    k = m.kind("k")
    assert k.flag("F_A", 0) == 1 and k.flag("F_B", 12) == 1 << 12
    with pytest.raises(LayoutError, match="bit 12"):
        k.flag("F_C", 12)
    assert k.flag("B_A", 12, of="format") == 1 << 12  # (another field)
    assert k.values("K_X K_Y", of="kind") == (0, 1)
    assert k.values("K_Z", of="kind", start=4) == (4,)
    with pytest.raises(LayoutError, match="value 1"):
        k.values("K_W", of="kind", start=1)
    w = k.word("W", at=3)
    assert k.alias("W2", w) == 3
    with pytest.raises(LayoutError, match="no word"):
        k.alias("W3", 9)
    assert k.consts() == {"W": 3, "W2": 3, "F_A": 1, "F_B": 1 << 12, "B_A": 1 << 12, "K_X": 0, "K_Y": 1, "K_Z": 4}


def test_consts_compose_without_clashes() -> None:
    m = WordMap("m", 16, objects=True)
    m.common.word("H", at=0)
    a, b, c = m.kind("a"), m.kind("b"), m.kind("c")
    a.word("X", at=4)
    a.flag("F_A", 3)
    b.word("X", at=4)  # (the same name and value in two kinds)
    b.flag("F_B", 4)
    c.word("X", at=5)
    c.flag("F_C", 3)
    assert m.consts(a, b) == {"H": 0, "X": 4, "F_A": 8, "F_B": 16}
    with pytest.raises(LayoutError, match="X is 4 and 5"):
        m.consts(a, c)
    d = m.kind("d")
    d.flag("F_D", 3)
    with pytest.raises(LayoutError, match="F_A and F_D take one bit of flags"):
        m.consts(a, d)


def test_layout_table() -> None:
    m = WordMap("m", 8, objects=True)
    m.common.words("H0 H1", at=0)
    m.kind("k").word("X", at=4)
    assert m.layout() == [("common", "H0", 0, 1), ("common", "H1", 1, 1), ("k", "X", 4, 1)]
