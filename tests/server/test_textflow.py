"""The generated text's filters (``models.server.textflow``) on the host: the thought split (by markers, by blocks),
a thought's budget, the stop strings and the text held back until what follows decides."""

from models.server.textflow import BlockSplitter, ReasoningSplitter, StopMatcher, ThoughtBudget, held

from .fakes import BLOCKS, GEMMA, QWEN


def test_held() -> None:
    assert held("abc ST", ["STOP"]) == 2
    assert held("abc", ["STOP"]) == 0
    assert held("STO", ["STOP", "TOO"]) == 3
    assert held("STOP", ["STOP"]) == 0  # (all of a word is no start of one)


def test_splitter_and_stops() -> None:
    s = ReasoningSplitter(QWEN, inside=True)
    parts = [p for t in ["\nA", " b\n", "\n</th", "ink>", "\n", "\nC", "\n\nD", "\n"] for p in s.feed(t)] + s.flush()
    assert parts == [("reasoning_content", "A"), ("reasoning_content", " b"), ("content", "C"), ("content", "\n\nD")]
    s = ReasoningSplitter(GEMMA)
    parts = [p for t in ["Plain <|chan", "nel", " not a marker"] for p in s.feed(t)] + s.flush()
    assert "".join(t for _, t in parts) == "Plain <|channel not a marker" and {k for k, _ in parts} == {"content"}
    # a Qwen call that ends the thought without its close marker; a thought only begins the answer (Qwen's)
    s = ReasoningSplitter(QWEN, inside=True, ends=("<tool_call>",), once=True)
    parts = [p for t in ["Need to look.\n<tool", "_call>\n<function=f>", " text </think> <think> more"]
             for p in s.feed(t)] + s.flush()
    assert parts[0] == ("reasoning_content", "Need to look.") and {k for k, _ in parts[1:]} == {"content"}
    assert "".join(t for _, t in parts[1:]) == "<tool_call>\n<function=f> text </think> <think> more"
    s = ReasoningSplitter(QWEN, once=True)
    assert s.feed("</think>\n\nA <think>") + s.flush() == [("content", "A <think>")]
    m = StopMatcher(["STOP", "\n\n"])
    assert [m.feed(t) for t in ["Hello", " wor", "ld ST", "O"]] == [("Hello", False), (" wor", False), ("ld ", False),
                                                                     ("", False)]
    assert m.feed("P more") == ("", True) and m.word == "STOP"
    m = StopMatcher(["STOP"])
    assert m.feed("ST") == ("", False) and m.feed("ART") == ("START", False)


def test_blocks() -> None:
    """A turn of blocks, whatever its pieces: the thoughts, the text blocks' content without their markers or the white
    space around their text; another block (a call's) as it is; the thought the prompt opened; a budget counted inside
    thought blocks alone."""
    H, TH, TX, C, E = BLOCKS.header, BLOCKS.thought, BLOCKS.text, BLOCKS.call, BLOCKS.end
    text = f"{H}{TH} Hmm,\nyes. \n{E}{H}{TX}\nA {E}{H}{TX}B{E}{H}f{C}{{}}{E}"
    for step in (1, 2, 5, len(text)):
        s = BlockSplitter(BLOCKS)
        parts = [p for i in range(0, len(text), step) for p in s.feed(text[i:i + step])] + s.flush()
        joined = {k: "".join(v for kk, v in parts if kk == k) for k in ("reasoning_content", "content")}
        assert joined == {"reasoning_content": "Hmm,\nyes.", "content": f"AB{H}f{C}{{}}{E}"}, step
    s = BlockSplitter(BLOCKS, inside=True)
    assert s.inside and s.feed(f"still{E}") == [("reasoning_content", "still")] and s.answering and not s.inside
    b = ThoughtBudget(BlockSplitter(BLOCKS), budget=2)
    for piece, due in [(H, False), (TH, False), ("a", False), (" b", True)]:
        b.feed(piece)
        assert b.due() is due, piece
    b = ThoughtBudget(BlockSplitter(BLOCKS), budget=1)
    for piece in (H, TX, "a", " b"):
        b.feed(piece)
        assert not b.due()  # (a text block: no thought)
