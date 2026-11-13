"""Decisions on the NPU (``models.server.decisions``; vllm-d3's d3-lite: qwen35, non-causal, every tensor Q8_0): each
question's prompt through ``NpuModel.outputs`` (one chunk from position 0), its label logits on the host, against the same
label rows times llama.cpp's last row of the same prompt (``goldens/qwen35_dump.cpp``'s ``result_norm``: the whole prompt in
one graph, the file's non-causal attention), the answers' probabilities within 0.03 and the same choices. llama.cpp's
CPU and Vulkan builds are 0.019 apart at most on 320 such questions, the NPU 0.087 (median 0.003): the prefill's bfp16
GEMMs, ~3% of each layer's own output. vllm-decision2's Decision-2.0-Kai-0.6B (qwen3 on the Llama path, its joint head on
the host) against the fp32 CPU model's rows (``llama/cpu.py``) through the same head: within 0.05 (against llama-server's
CPU answers on 320 questions: 0.040 at most, 313 of 320 choices the same). LiquidAI's d1-2B (lfm3, its full layers
non-causal): each question alone against llama.cpp's last row, a state's questions in one sequence (its ``set``
attention) against the fp32 CPU model's rows (``lfm/cpu.py``) of the same sequence, within 0.03 and the same choices.
Requests with images (d1-2B, d3-lite, d3-nano with their mmproj: ``test_images``): each question alone against
llama-server's CPU answers (cached), the same usage; d1's sequences against the fp32 CPU model on the same image rows,
d3's packed chunks of image prompts against each prompt alone."""

import base64
import io
import json
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from artlite.model_config import decision_gguf, decision_mmproj
from models import open_model
from models.server.decisions import Decisions

from .common import cached_ref, essay, model_key
from .goldens.text import golden
from .llama_cpp import lib_identity, llama_server_decisions
from .models import require

pytestmark = pytest.mark.slow  # GGUF models on the NPU: run with --run-slow

QUESTIONS = {
    "topic": {"type": "choice", "instructions": "What is the passage mostly about?",
              "criteria": {"programming": None, "cooking": None, "sports": None, "music": None}},
    "tone": {"type": "score", "instructions": "How optimistic is the tone of the passage?",
             "criteria": ["very pessimistic", "pessimistic", "neutral", "optimistic", "very optimistic"]},
    "advice": {"type": "noul", "instructions": "Does the passage give advice to the reader?"},
}


def test_d3_lite() -> None:
    path = decision_gguf("d3-lite")
    require(path)
    model = open_model(path)
    npu = model.load(capacity=4096)
    assert not npu.causal
    d = Decisions.of(path, model.tokenizer.encode)
    text = essay("apple.txt")
    worst = 0.0
    for state in (text[:400], text[2000:5000]):  # (~170 and ~800 prompt tokens: chunks of 256 and 1024)
        body = {"state": state, "questions": QUESTIONS}
        qs = d.questions(body)
        for p in d.prompts(state, qs, model.tokenizer.encode):
            q = qs[p.question]
            npu.reset()
            got = d.answer(q, [d.scores(q, npu.outputs(p.ids, p.rows))])
            g = golden(path, p.ids, layers=("g",), names="result_norm", logits="none")
            want = d.answer(q, [d.scores(q, np.asarray(g["result_norm"], np.float64).reshape(-1, model.cfg.hidden_size))])
            assert got.get("choice") == want.get("choice"), (q.id, got, want)
            pw, pg = (a.get("probabilities", {"true": a.get("noul")}) for a in (want, got))
            worst = max(worst, *(abs(pw[k] - pg[k]) for k in pw))
    print(f"max |dp| {worst:.4f}")
    assert worst < 0.03


def test_kai() -> None:
    from artlite.gguf_model import GGUFFile

    from .llama.cpu import LlamaCpu

    path = decision_gguf("kai")
    require(path)
    model = open_model(path)
    npu = model.load(capacity=4096)
    d = Decisions.of(path, model.tokenizer.encode)
    cpu = LlamaCpu(GGUFFile(path))
    worst = 0.0
    state = essay("apple.txt")[:1200]
    qs = d.questions({"state": state, "questions": QUESTIONS})
    for p in d.prompts(state, qs, model.tokenizer.encode):
        q = qs[p.question]
        npu.reset()
        got = d.answer(q, [d.scores(q, npu.outputs(p.ids, p.rows))])
        want = d.answer(q, [d.scores(q, cpu.outputs(p.ids, list(p.rows)).numpy())])  # (rows ascending: options, question)
        pw, pg = (a.get("probabilities", {"true": a.get("noul")}) for a in (want, got))
        worst = max(worst, *(abs(pw[k] - pg[k]) for k in pw))
    print(f"max |dp| {worst:.4f}")
    assert worst < 0.05


def test_d1() -> None:
    from artlite.gguf_model import GGUFFile

    from .lfm.cpu import LfmCpu

    path = decision_gguf("d1-2b")
    require(path)
    model = open_model(path)
    npu = model.load(capacity=4096)
    assert not npu.causal
    d = Decisions.of(path, model.tokenizer.encode)
    cpu = LfmCpu(GGUFFile(path), vocab=0)
    text = essay("apple.txt")
    worst = {1: 0.0, 16: 0.0}
    for state in (text[:400], text[2000:5000]):  # (~130 and ~750 tokens a question alone)
        qs = d.questions({"state": state, "questions": QUESTIONS})
        for size in (1, 16):
            for p in d.prompts(state, qs, model.tokenizer.encode, set_size=size):
                npu.reset()
                rows = npu.outputs(p.ids, p.rows, sets=p.sets)
                if size == 1:  # (llama.cpp reads each question alone)
                    g = golden(path, p.ids, layers=("g",), names="result_norm", logits="none")
                    want = np.asarray(g["result_norm"], np.float64).reshape(-1, model.cfg.hidden_size)[-1:]
                else:
                    want = cpu.outputs(p.ids, list(p.rows), sets=p.sets).double().numpy()
                assert len(p.questions) == (1 if size == 1 else len(qs))
                for j, i in enumerate(p.questions):
                    q = qs[i]
                    a, b = (d.answer(q, [d.scores(q, x[j:j + 1])]) for x in (rows, want))
                    assert a.get("choice") == b.get("choice"), (q.id, a, b)
                    pa, pb = (x.get("probabilities", {"true": x.get("noul")}) for x in (a, b))
                    worst[size] = max(worst[size], *(abs(pa[k] - pb[k]) for k in pa))
    print(f"max |dp|: alone vs llama.cpp {worst[1]:.4f}, a sequence vs the CPU model {worst[16]:.4f}")
    assert max(worst.values()) < 0.03


ASTRONAUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "sd15-astronaut.jpg")
IMAGE_QUESTIONS = {
    "scene": {"type": "choice", "instructions": "What does the image mainly show?",
              "criteria": {"animal": None, "vehicle": None, "person": None, "text document": None,
                           "abstract shapes": None}},
    "person": {"type": "noul", "instructions": "Is there a person in the image?"},
    "count": {"type": "score", "instructions": "How many separate objects are in the image?",
              "criteria": ["none", "one", "two", "three", "four or more"]},
    "color": {"type": "choice", "instructions": "What is the dominant color of the main object?",
              "criteria": {"red": None, "green": None, "blue": None, "white": None}},
    "text": {"type": "noul", "instructions": "Does the image contain readable text?"},
}
# (the largest |dp| from llama-server's CPU answers, each question alone: the NPU's prefill GEMMs, and for d3 the vision
# overlay's bfp16 encoder (its 24 layers of 1024: ~7% per merged token, as the CPU at its rounding points); on 57 such
# questions d1 Q4_0 0.021, d3-lite 0.117, d3-nano 0.097)
IMAGE_DP = {"d1-2b": 0.05, "d3-lite": 0.12, "d3-nano": 0.12}


def _url(im: Image.Image, fmt: str) -> str:
    buf = io.BytesIO()
    im.save(buf, fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def image_bodies() -> list[dict]:
    """The astronaut (512 x 512: 256 tokens) with a state; a drawn card (a red circle, three blue squares, a line of
    text) and the astronaut at 384 x 384 with a null state; a chat state with the card in an image_url part."""
    card = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(card)
    d.ellipse((60, 60, 260, 260), fill=(220, 30, 30))
    for x in (320, 420, 520):
        d.rectangle((x, 120, x + 70, 190), fill=(30, 60, 210))
    d.text((60, 340), "TOTAL DUE: $1,250.00", fill=(0, 0, 0), font=ImageFont.load_default(size=40))
    with Image.open(ASTRONAUT) as im:
        astro = im.convert("RGB")
    a, c = _url(astro, "JPEG"), _url(card, "PNG")
    small = _url(astro.resize((384, 384), Image.Resampling.BICUBIC), "JPEG")
    return [{"state": "A listing photo a seller submitted.", "images": [a], "questions": IMAGE_QUESTIONS},
            {"state": None, "images": [c, small], "questions": IMAGE_QUESTIONS},
            {"state": [{"role": "user", "content": [{"type": "text", "text": "What do I owe?"},
                                                    {"type": "image_url", "image_url": {"url": c}}]}],
             "questions": {k: IMAGE_QUESTIONS[k] for k in ("text", "count", "color")}}]


def _probs(a: dict) -> dict[str, float]:
    return a.get("probabilities", {"true": a.get("noul")})


def _top(a: dict) -> str:
    p = _probs(a)
    return str(p["true"] >= 0.5) if a["type"] == "noul" else max(p, key=p.get)


@pytest.mark.parametrize("name", ["d1-2b", "d3-lite", "d3-nano"])
def test_images(name: str) -> None:
    path, mmproj = decision_gguf(name), decision_mmproj(name)
    if not (os.path.exists(path) and os.path.exists(mmproj)):
        pytest.skip(f"{path} or {mmproj} missing")
    bodies = image_bodies()
    args = ("--image-max-tokens", "1600") if name == "d1-2b" else ()  # (Artlite's d1 budget: its runner's)
    want = cached_ref(path, "systemone_images", (model_key(path), model_key(mmproj), json.dumps(bodies), args,
                                                 lib_identity()),
                      lambda: llama_server_decisions(path, mmproj, bodies, args))
    model = open_model(path, mmproj=mmproj)
    npu = model.load(capacity=4096)
    d = Decisions.of(path, model.tokenizer.encode)
    enc = model.tokenizer.encode
    whole = npu.prefill.sizes[-1]
    cpu = None
    if name == "d1-2b":
        from artlite.gguf_model import GGUFFile

        from .lfm.cpu import LfmCpu
        cpu = LfmCpu(GGUFFile(path), vocab=0)
    worst, same, n, worst_set = 0.0, 0, 0, 0.0
    for body, ref_out in zip(bodies, want, strict=True):
        qs = d.questions(body)
        state, images = d.state(body)
        media = [model.image_tokens(b) for b in images]
        lone = d.prompts(state, qs, enc, media=media, set_size=1, max_tokens=whole)
        assert sum(len(p.ids) for p in lone) == ref_out["usage"]["input_tokens"]
        rows = []
        for p in lone:
            npu.reset()
            rows.append(npu.outputs(p.ids, p.rows, **({"sets": p.sets} if p.sets else {})))
        for p, r in zip(lone, rows, strict=True):
            q = qs[p.question]
            got, ref_a = d.answer(q, [d.scores(q, r)], media=True), ref_out["answers"][q.id]
            same += _top(got) == _top(ref_a)
            n += 1
            worst = max(worst, *(abs(_probs(got)[k] - _probs(ref_a)[k]) for k in _probs(ref_a)))
        if cpu is not None:  # (one sequence of the state's questions: the runner's set attention, against the CPU model)
            (p,) = d.prompts(state, qs, enc, media=media, max_tokens=whole)
            npu.reset()
            got = npu.outputs(p.ids, p.rows, sets=p.sets)
            exact = cpu.outputs(p.ids, list(p.rows), sets=p.sets).double().numpy()
            for j, i in enumerate(p.questions):
                q = qs[i]
                a, b = (d.answer(q, [d.scores(q, x[j:j + 1])], media=True) for x in (got, exact))
                assert _top(a) == _top(b), (q.id, a, b)
                worst_set = max(worst_set, *(abs(_probs(a)[k] - _probs(b)[k]) for k in _probs(a)))
        else:  # (several image prompts in packed chunks, against each alone)
            npu.reset()
            packed = npu.readouts([p.ids for p in lone], [p.rows for p in lone])
            for p, r, g in zip(lone, rows, packed, strict=True):
                q = qs[p.question]
                a, b = (d.answer(q, [d.scores(q, x)], media=True) for x in (g, r))
                worst_set = max(worst_set, *(abs(_probs(a)[k] - _probs(b)[k]) for k in _probs(a)))
    print(f"{name}: {same} / {n} same top choice as llama-server CPU, max |dp| {worst:.4f}; "
          f"{'a sequence vs the CPU model' if cpu else 'packed vs alone'} {worst_set:.4f}")
    assert same >= n - 1 and worst < IMAGE_DP[name]
    assert worst_set < 0.03


AUDIO_QUESTIONS = {
    "voice": {"type": "choice", "instructions": "Who is speaking?",
              "criteria": {"a man": None, "a woman": None, "a child": None}},
    "topic": {"type": "choice", "instructions": "What is the speaker talking about?",
              "criteria": {"a meal": None, "a sports game": None, "a computer program": None, "a celebration": None}},
    "calm": {"type": "noul", "instructions": "Does the speaker sound calm?"},
    "urgency": {"type": "score", "instructions": "How urgent is the message?",
                "criteria": ["not urgent", "somewhat urgent", "urgent", "very urgent"]},
}


def omni_bodies(clip: bytes, question: bytes, video: bytes) -> list[dict]:
    """A clip in ``files`` with a state; the astronaut with a spoken question (a LibriSpeech clip as its instructions)
    and a written one, a null state; a chat state with a video's frames (``video_url``) and text."""
    b64 = lambda b: base64.b64encode(b).decode()
    with Image.open(ASTRONAUT) as im:
        astro = _url(im.convert("RGB"), "JPEG")
    spoken = {"type": "choice", "instructions": {"type": "input_audio", "input_audio": {"data": b64(question)}},
              "criteria": {"an animal": None, "a vehicle": None, "a building": None}}
    return [{"state": "A voicemail a customer left.", "files": ["data:audio/wav;base64," + b64(clip)],
             "questions": AUDIO_QUESTIONS},
            {"state": None, "images": [astro], "questions": {"spoken": spoken, "person": IMAGE_QUESTIONS["person"]}},
            {"state": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,"
                                                                                       + b64(video)}},
                                                    {"type": "text", "text": "Does this look right?"}]}],
             "questions": {k: IMAGE_QUESTIONS[k] for k in ("scene", "person", "count")}}]


def test_omni_media(tmp_path: object) -> None:
    """d1-Omni-2B's media (``lfm3-d1-omni``): an audio clip, a spoken question (heard where its text would be, after
    the state's media), a video's frames; each question alone against llama-server's CPU answers (cached), the same
    usage."""
    from .common import SPEECH_CLIP, video_file
    path, mmproj = decision_gguf("d1-omni"), decision_mmproj("d1-omni")
    if not (os.path.exists(path) and os.path.exists(mmproj) and os.path.exists(SPEECH_CLIP)):
        pytest.skip(f"{path}, {mmproj} or {SPEECH_CLIP} missing")
    with open(SPEECH_CLIP, "rb") as f:
        clip = f.read()
    with open(SPEECH_CLIP.replace("0001", "0000"), "rb") as f:
        question = f.read()
    with open(video_file(str(tmp_path), "astronaut-small"), "rb") as f:
        video = f.read()
    bodies = omni_bodies(clip, question, video)
    args = ("--image-max-tokens", "1600")  # (Artlite's d1 budget: its runner's)
    want = cached_ref(path, "systemone_omni", (model_key(path), model_key(mmproj), json.dumps(bodies), args,
                                               lib_identity()),
                      lambda: llama_server_decisions(path, mmproj, bodies, args))
    model = open_model(path, mmproj=mmproj)
    npu = model.load(capacity=4096)
    d = Decisions.of(path, model.tokenizer.encode)
    enc = model.tokenizer.encode
    from models.server.decisions import spoken
    worst, same, n = 0.0, 0, 0
    for body, ref_out in zip(bodies, want, strict=True):
        qs = d.questions(body)
        state, files = d.state(body)
        media = [model.image_tokens(b) if b.kind == "image" else model.media_tokens(b, b.kind) for b in files]
        heard = {i: model.media_tokens(base64.b64decode(q.instructions["input_audio"]["data"]), "audio")
                 for i, q in enumerate(qs) if spoken(q.instructions)}
        lone = d.prompts(state, qs, enc, media=media, set_size=1, max_tokens=npu.prefill.sizes[-1], spoken_media=heard)
        assert sum(len(p.ids) for p in lone) == ref_out["usage"]["input_tokens"]
        for p in lone:
            q = qs[p.question]
            npu.reset()
            got, ref_a = d.answer(q, [d.scores(q, npu.outputs(p.ids, p.rows, sets=p.sets))], media=True), \
                ref_out["answers"][q.id]
            same += _top(got) == _top(ref_a)
            n += 1
            worst = max(worst, *(abs(_probs(got)[k] - _probs(ref_a)[k]) for k in _probs(ref_a)))
            print(q.id, _top(got), _top(ref_a), {k: round(v, 3) for k, v in _probs(got).items()})
    print(f"d1-omni: {same} / {n} same top choice as llama-server CPU, max |dp| {worst:.4f}")
    assert same >= n - 1 and worst < 0.06
