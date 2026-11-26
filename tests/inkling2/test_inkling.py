"""Inkling 2's host side (no NPU): the configuration and its visit table (``model_config.InklingConfig``; unsloth's root
layout read as the loop one), the companion files' discovery, the tokenizer against llama.cpp's ``llama-tokenize`` (the
whole essay corpus, the chat renders with the special tokens), the chat template against transformers' renders
(``renders.json``: transformers 5.24.0's ``apply_chat_template`` renderer on the file's template: tools and their
calls, reasoning efforts, thoughts, images, content parts) and ``Inkling.chat``'s reasoning effort."""

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from artlite.gguf_model import GGUFFile, QTensor
from artlite.model_config import InklingConfig, inkling_dspark_of, inkling_gguf, inkling_mmproj_of
from models import open_model
from models.inkling2.inkling import Inkling
from models.templating import render_chat

from ..common import ESSAYS_DIR, essay
from ..llama_cpp import llama_tokenize
from ..model_files import require

MODEL = inkling_gguf()
RENDERS = json.loads((Path(__file__).parent / "renders.json").read_text())["cases"]


@pytest.fixture(scope="module")
def model() -> Inkling:
    require(MODEL)
    m = open_model(MODEL)
    assert isinstance(m, Inkling)
    return m


def test_config() -> None:
    """The shapes, the 48 visits (their layers, passes, llama.cpp's cache slots, kinds, blocks, mix queries), the
    loop tensors' names, tau, and a DSpark drafter's taps as visits."""
    require(MODEL)
    g = GGUFFile(MODEL)
    c = InklingConfig.from_gguf(g)
    assert (c.hidden_size, c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.head_dim) == \
        (2304, 24, 24, 8, 128)
    assert (c.vocab_size, c.vocab_padded, c.n_experts, c.n_experts_used, c.expert_ffn) == (200058, 201024, 128, 8, 1152)
    assert (c.core_start, c.core_end, c.passes, c.gate_rank, c.block_size, c.layout) == (6, 18, 3, 64, 6, "loop")
    assert c.layers("full") == [5, 11, 17, 23] and c.sliding_window == c.rel_extent == 1024
    assert c.eos_token_ids == (200006, 199999) and c.softcap == 30 and c.expert_weights_scale == 8
    v = c.visits
    assert [x.layer for x in v] == [*range(6), *list(range(6, 18)) * 3, *range(18, 24)]
    assert [x.loop for x in v] == [-1] * 6 + [p for p in range(3) for _ in range(12)] + [-1] * 6
    assert sorted(x.slot for x in v) == list(range(48)) and (v[18].slot, v[41].slot, v[42].slot) == (24, 47, 18)
    assert [x.index for x in v if x.kind == "full"] == [5, 11, 17, 23, 29, 35, 41, 47]
    assert [x.index for x in v if x.messages] == list(range(18, 42))
    assert [x.index for x in v if x.block] == [0, 6, 12, 18, 24, 30, 36, 42]
    assert [x.queries for x in v[:7]] == [0, 1, 1, 1, 1, 3, 1]
    for layer in c.layers("core"):
        assert c.msg_queries(g, layer).shape == (4, 2304)
        for s in c.CTRL_STREAMS:
            assert [g.tensor(c.ctrl_tensor(layer, s, p)).cols for p in ("down", "gain", "forget")] == [4608, 64, 64]
    t = c.tau([0, 8190, 8191, 16383, 1 << 20])
    assert t.dtype == np.float32 and list(t[:3]) == [1, 1, 1]
    assert np.allclose(t[3:], [1 + 0.1 * math.log(2), 1 + 0.1 * math.log(128)], rtol=1e-6)
    d = inkling_dspark_of(MODEL)
    assert d is not None
    taps = [int(x) for x in GGUFFile(d).arch_meta("target_layers")]
    assert taps[-1] == len(v) and [(v[i].layer, v[i].loop) for i in taps[:-1]] == [(5, -1), (11, 1), (17, 2)]


class RootView:
    """The loop file as unsloth's root files spell it (GGUF ``inkling2``: 48 blocks, ``num_loops``, ``loop.core_*``,
    48-entry arrays; the controllers ``loop_{attn,k,v,mlp}_*``; the message queries in two tables ``loop_res_query``
    (q, ffn) and ``loop_kv_res_query`` (k, v), rows 2 i and 2 i + 1 for core layer i, as llama.cpp's converter reads
    the checkpoint's)."""

    CTRL: ClassVar[dict[str, str]] = {"attn": "q", "k": "k", "v": "v", "mlp": "ffn"}

    def __init__(self, g: GGUFFile, c: InklingConfig):
        self.g, self.c, self.path, self.arch = g, c, g.path, "inkling2"
        rename = {"loop.start": "loop.core_start", "loop.end": "loop.core_end", "loop.count": "num_loops"}
        slots = [x.layer for x in sorted(c.visits, key=lambda x: x.slot)]
        self.meta: dict[str, object] = {}
        for k, val in g.meta.items():
            if not k.startswith("inkling2_loop."):
                self.meta[k] = val
                continue
            k = k.removeprefix("inkling2_loop.")
            if k == "loop.varying":
                continue
            if isinstance(val, np.ndarray) and len(val) == c.num_hidden_layers:
                val = val[slots]
            self.meta["inkling2." + rename.get(k, k)] = len(slots) if k == "block_count" else val
        self.tables = {"loop_res_query.weight": (0, 3), "loop_kv_res_query.weight": (1, 2)}

    def arch_meta(self, key: str, default: object = None) -> object:
        return self.meta.get(f"inkling2.{key}", default)

    def string_count(self, key: str) -> int:
        return self.g.string_count(key)

    def strings(self, key: str, first: int = 0) -> list[str]:
        return self.g.strings(key, first)

    def _loop_name(self, name: str) -> str:
        parts = name.split(".")
        if len(parts) == 4 and parts[2].startswith("loop_"):
            stream, part = parts[2].removeprefix("loop_").rsplit("_", 1)
            return f"{parts[0]}.{parts[1]}.loop_{self.CTRL[stream]}_ctrl_{part}.weight"
        return name

    def __contains__(self, name: str) -> bool:
        return name in self.tables or self._loop_name(name) in self.g

    def tensor(self, name: str) -> QTensor:
        if name in self.tables:
            q = [self.c.msg_queries(self.g, layer) for layer in self.c.layers("core")]
            rows = np.stack([q[i][j] for i in range(len(q)) for j in self.tables[name]])
            return QTensor(name=name, qtype="F32", rows=rows.shape[0], cols=rows.shape[1], raw=rows)
        return self.g.tensor(self._loop_name(name))


def test_root_layout() -> None:
    """unsloth's root layout (``inkling2`` with a loop) gives the loop file's configuration, message queries and
    controller weights."""
    require(MODEL)
    g = GGUFFile(MODEL)
    c = InklingConfig.from_gguf(g)
    view = RootView(g, c)
    r = InklingConfig.from_gguf(view)
    assert r.layout == "root" and dataclasses.replace(r, layout="loop") == c
    for layer in c.layers("core"):
        assert np.array_equal(r.msg_queries(view, layer), c.msg_queries(g, layer))
        for s in c.CTRL_STREAMS:
            name = r.ctrl_tensor(layer, s, "gain")
            assert name in view and "ctrl" not in name
            assert np.array_equal(view.tensor(name).raw, g.tensor(c.ctrl_tensor(layer, s, "gain")).raw)


def test_files(tmp_path: Path) -> None:
    """The drafter and the vision tower beside the model, or in the repository's root above ``Mainline_Support`` (Q8_0 /
    F16 first, ``dspark/`` after); a file that is not one (another projector, another vocabulary) is not taken."""
    require(MODEL)
    d, mm = inkling_dspark_of(MODEL), inkling_mmproj_of(MODEL)
    assert os.path.basename(d) == "dspark-Inkling-2-Mini-Q8_0.gguf" and os.path.basename(mm) == "mmproj-F16.gguf"
    repo = tmp_path / "unsloth" / "Inkling-2-Mini-GGUF"
    (repo / "Mainline_Support").mkdir(parents=True)
    (repo / "dspark").mkdir()
    model = repo / "Mainline_Support" / os.path.basename(MODEL)
    model.symlink_to(MODEL)
    assert inkling_dspark_of(str(model)) is None and inkling_mmproj_of(str(model)) is None
    (repo / "dspark" / "dspark-Inkling-2-Mini-BF16.gguf").symlink_to(d)
    (repo / "mmproj-BF16.gguf").symlink_to(mm)
    assert inkling_dspark_of(str(model)) == str(repo / "dspark" / "dspark-Inkling-2-Mini-BF16.gguf")
    assert inkling_mmproj_of(str(model)) == str(repo / "mmproj-BF16.gguf")
    (repo / "dspark-Inkling-2-Mini-Q8_0.gguf").symlink_to(d)
    (repo / "mmproj-F16.gguf").symlink_to(mm)
    (repo / "mmproj-Other.gguf").symlink_to(d)  # (a drafter, no mmproj)
    assert inkling_dspark_of(str(model)) == str(repo / "dspark-Inkling-2-Mini-Q8_0.gguf")
    assert inkling_mmproj_of(str(model)) == str(repo / "mmproj-F16.gguf")


def test_corpus_tokens(model: Inkling) -> None:
    """The whole essay corpus in one text (159502 tokens): llama.cpp's ids, and decoded back."""
    names = sorted(n for n in os.listdir(ESSAYS_DIR) if n.endswith(".txt")) if os.path.isdir(ESSAYS_DIR) else []
    if not names:
        pytest.skip("no essay corpus (tests/datasets/fetch_datasets.py)")
    text = "".join(essay(n) for n in names)
    ids = model.encode(text)
    assert ids == llama_tokenize(MODEL, text)
    assert model.tokenizer.decode(ids) == text


@pytest.mark.parametrize("case", list(RENDERS))
def test_renders(model: Inkling, case: str) -> None:
    """The file's template renders transformers' text, and its tokens (every special token of the template) are
    llama.cpp's."""
    c = RENDERS[case]
    text = render_chat(model.gguf, c["messages"], tools=c["tools"], **c["variables"])
    assert text == c["text"]
    assert model.tokenizer.encode(text) == llama_tokenize(MODEL, text)


def test_chat_effort(model: Inkling) -> None:
    """``Inkling.chat``: thinking at the template's effort (default 0.9, or the given level), else effort none unless
    the variables give one; earlier thoughts only with ``preserve``."""
    msgs, think = RENDERS["basic"]["messages"], RENDERS["thought_kept"]["messages"]
    assert model.chat_text(msgs, think=True) == RENDERS["basic"]["text"]
    assert model.chat_text(msgs, think=False) == RENDERS["effort_none"]["text"]
    for level in ("low", "max"):
        for t in (False, True):
            assert model.chat_text(msgs, think=t, variables={"reasoning_effort": level}) == \
                RENDERS[f"effort_{level}"]["text"]
    kept = model.chat_text(think, think=True, preserve=True, variables={"reasoning_effort": "high"})
    assert kept == RENDERS["thought_kept"]["text"] and "<|content_thinking|>" in kept
    assert "<|content_thinking|>" not in model.chat_text(think, think=True)
    ids = model.chat(msgs)
    assert ids == llama_tokenize(MODEL, RENDERS["effort_none"]["text"]) and ids[0] == 200002  # (no BOS)


def test_blocks(model: Inkling) -> None:
    """The turn the model writes as blocks (``Model.blocks``): its markers are text in a generation (llama.cpp's
    preserved tokens; another control token is not, and ``decode(special=True)`` writes it), one token each; a prompt
    opens no block, so the model thinks first at any effort (``structured.prompt_thinks``), but where it continues the
    answer's text; the prompt cache's mark of the first user message (``user_turn``)."""
    from models.structured import prompt_thinks
    b = model.blocks
    turn = f"{b.header}{b.thought}Hmm{b.end}{b.header}{b.text}Yes{b.end}"
    ids = model.tokenizer.encode(turn + "<|content_model_end_sampling|>")
    assert len(ids) == 9 and model.tokenizer.decode(ids) == turn
    assert model.tokenizer.decode(ids, special=True) == turn + "<|content_model_end_sampling|>"
    for think in (False, True):
        prompt = model.chat(RENDERS["basic"]["messages"], think=think)
        assert prompt_thinks(model.tokenizer, prompt, model.reasoning, think, blocks=b)
        assert not prompt_thinks(model.tokenizer, prompt + model.tokenizer.encode(b.answer), model.reasoning, think,
                                 blocks=b)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    text = model.chat_text([{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}], tools=tools)
    assert text.count(model.user_turn) == 1 and text.index(model.user_turn) > text.index("tool_declare")
