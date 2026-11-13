"""The generated text's filters (``models.server.textflow``) on the host: the thought split, the stop strings and the
text held back until what follows decides."""

from models.server.textflow import ReasoningSplitter, StopMatcher, held

from .fakes import GEMMA, QWEN


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
