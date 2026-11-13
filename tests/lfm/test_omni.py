"""LFM3-Omni-2B (models.lfm with LFM3-Omni's mmproj: its vision tower, speech encoder and speech generator).

On the NPU (slow; the tower, the speech encoder and the generator on one device: ``models.lfm.lfm.media_device``):
teacher-forced over llama.cpp's answers to a picture, a clip (ASR) and a picture with a spoken question (Omni's own
speech), the KL against llama.cpp's language model on float64 embeddings within 2x of the NPU language model's own; a
TTS run that ends with the end of audio and that the model's ASR hears back; a voice reply about a picture.

Host (no NPU): the prompts' token ids are llama.cpp's (libmtmd's chunks through ``tests.minicpmv.ref.golden``: text ids,
each media chunk's tokens) for an image, an audio clip, an image with a spoken question, videos with and without a
soundtrack (Omni's layout: per pair its timestamp, its tokens, its piece of the soundtrack), an LFM3-VL-2B video (frames
only); the TTS and interleaved prompts' ids are llama.cpp's tokenization of their text. The encoders here are stand-ins
that give the real token counts (``models.lfm3vl.host``, ``models.lfm3a.host``) and zero rows."""

import os
import re
import time
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import pytest

from artlite.gguf_model import GGUFFile
from artlite.model_config import ClipConfig, LfmConfig, lfm3omni_gguf, lfm3omni_mmproj, lfm3vl_gguf, lfm3vl_mmproj
from models import video as VD
from models.lfm.lfm import Lfm
from models.lfm3a import host as A
from models.lfm3a import speech as S
from models.lfm3vl import host as Q
from models.sequence import MEDIA, Sequence, load_image
from models.tokenizer import BpeTokenizer, chat_prompt

from ..common import ASTRONAUT, SPEECH_CLIP, video_file
from ..lfm3a import ref as AR
from ..lfm3vl import ref as VR
from ..llama_cpp import llama_tokenize
from ..minicpmv.ref import golden

MODEL, MMPROJ = lfm3omni_gguf(), lfm3omni_mmproj()
MARKER = "<__media__>"


class _Vision:
    """The vision tower's token counts (``NpuVision``'s bounds and sizes), zero rows."""

    def __init__(self, mmproj: str, hidden: int):
        c = ClipConfig.from_gguf(GGUFFile(mmproj))
        self.min_tokens, self.max_tokens = Q.token_bounds(c.min_pixels, c.max_pixels)
        self.hidden = hidden

    def sequence(self, image: str) -> Sequence:
        px = Q.image_pixels(load_image(image), min_tokens=self.min_tokens, max_tokens=self.max_tokens)
        g = Q.Grid(px.shape[0] // 32, px.shape[1] // 32)
        return g.sequence(np.zeros((g.tokens, self.hidden), np.float32))

    def pairs(self, frames: Iterable[np.ndarray]) -> list[np.ndarray]:
        fr = list(frames)
        g = Q.Grid(fr[0].shape[0] // 32, fr[0].shape[1] // 32)
        return [np.zeros((g.tokens, self.hidden), np.float32) for _ in range(0, len(fr), 2)]


class _Audio:
    """The speech encoder's rows (``host.rows``), zero rows."""

    def __init__(self, hidden: int):
        self.hidden = hidden

    def embeddings(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((A.rows(A.frames(len(x))), self.hidden), np.float32)

    def sequence(self, clip: str | np.ndarray) -> Sequence:
        x = clip if isinstance(clip, np.ndarray) else A.load_audio(clip)
        return A.clip_sequence(self.embeddings(x))


def _host(model: str, mmproj: str, audio: bool) -> Lfm:
    """The model's host side (template, tokenizer, the prompt's layout) with stand-in encoders: no weights packed."""
    m = object.__new__(Lfm)
    m.gguf = GGUFFile(model)
    m.cfg = LfmConfig.from_gguf(m.gguf)
    m.tokenizer = BpeTokenizer(m.gguf)
    m.bos = int(m.gguf.meta["tokenizer.ggml.bos_token_id"])
    m.vision = _Vision(mmproj, m.cfg.hidden_size)
    m.audio = _Audio(m.cfg.hidden_size) if audio else None
    m._videos = OrderedDict()
    return m


def _text(m: Lfm, messages: list[dict[str, object]]) -> str:
    """The chat template's text of a conversation (``Lfm.chat``'s, before its media are spliced)."""
    bos = m.gguf.strings("tokenizer.ggml.tokens", m.bos)[0]
    return chat_prompt(m.gguf, messages, enable_thinking=False, bos_token=bos)


def _files(*paths: str) -> None:
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f"{p} missing")


@pytest.fixture(scope="module")
def videos(tmp_path_factory: pytest.TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp("videos"))


# case -> (the user turn's parts, the files in order: "image", "audio" or a video name)
CASES = {
    "image": [("image", ASTRONAUT), ("text", "Describe the image.")],
    "audio": [("audio", SPEECH_CLIP)],
    "image+audio": [("image", ASTRONAUT), ("audio", SPEECH_CLIP)],
    "video-sound": [("video", "astronaut"), ("text", "Describe what you see and hear.")],
    "video-silent": [("video", "silent"), ("text", "What happens in the video?")],
    "video-odd": [("text", "Two videos:"), ("video", "odd-25fps"), ("video", "ntsc-start")],
    "vl-video": [("video", "big-1080p"), ("text", "What happens in the video?")],  # (LFM3-VL-2B: frames only)
}


@pytest.mark.parametrize("case", list(CASES))
def test_prompt_tokens(case: str, videos: str) -> None:
    """Artlite's prompt (``Lfm.chat``) has llama.cpp's token ids: its text tokens where llama.cpp's are, a media row
    where each of llama.cpp's media chunks' tokens is."""
    vl = case.startswith("vl-")
    model, mmproj = (lfm3vl_gguf(), lfm3vl_mmproj()) if vl else (MODEL, MMPROJ)
    _files(model, mmproj, SPEECH_CLIP)
    m = _host(model, mmproj, audio=not vl)
    content, files, mtmd = [], [], []
    for kind, what in CASES[case]:
        if kind == "text":
            content.append({"type": "text", "text": what})
            mtmd.append({"type": "text", "text": what})
            continue
        path = video_file(videos, what) if kind == "video" else what
        content.append({"type": kind, kind: path})
        files.append(path)
        mtmd.append({"type": "text", "text": MARKER})
    seq = m.chat([{"role": "user", "content": content}])
    # llama-server's prompt: the parts' text with the media marker (no newline around it), the template rendering it
    text = "".join(p["text"] for p in mtmd)
    prompt = _text(m, [{"role": "user", "content": text}])
    gold = golden(model, mmproj, files, prompt)
    want = [t for c in gold.chunks for t in c.get("text", [MEDIA] * (c.get("image", 0) + c.get("audio", 0)))]
    got = [int(t) for t in seq.ids] if isinstance(seq, Sequence) else seq
    assert got == want


@pytest.mark.parametrize("voice, language", [("female", "en"), ("male", "de")])
def test_tts_prompt(voice: str, language: str) -> None:
    """Omni's TTS prompt: the file's ``tts`` template (``<think></think>`` before ``<|audio_start|>``), llama.cpp's ids."""
    _files(MODEL)
    g = GGUFFile(MODEL)
    text = S.tts_prompt(g, "Hello there.", language=language, voice=voice)
    assert text.endswith("<|im_start|>assistant\n<think></think><|audio_start|>")
    assert BpeTokenizer(g).encode(text) == llama_tokenize(MODEL, text)


def test_interleaved_prompt() -> None:
    """A voice reply's prompt: the conversation with ``speech.INTERLEAVED`` as its system turn, the assistant turn opened
    by ``<think></think>`` (the template's default) and ``<|mixed_start|>``: llama.cpp's ids of that text."""
    _files(MODEL, MMPROJ)
    m = _host(MODEL, MMPROJ, audio=True)
    msgs = [{"role": "system", "content": S.INTERLEAVED}, {"role": "user", "content": "What is the capital of France?"}]
    text = _text(m, msgs) + S.MIXED_START
    assert text.endswith("<|im_start|>assistant\n<think></think><|mixed_start|>")
    mixed = m.gguf.strings("tokenizer.ggml.tokens").index(S.MIXED_START)
    assert m.chat(msgs) + [mixed] == llama_tokenize(MODEL, text)


# ---------------------------------------------------------------------------------------------------------------------
# On the NPU (slow): the app's path (``Lfm(mmproj=)``: the tower, the speech encoder and the generator on one device)


IMAGE_ANSWER = ("The image is a surreal, digitally composed scene set in a barren, desert-like landscape. A "
                "chestnut/orange horse is captured mid-gallop, its body angled toward the left with one front leg lifted "
                "and its tail streaming behind.")  # (llama.cpp's greedy answer, CPU)
CLIP_ANSWER = ("The English forwarded to the French baskets of flowers, of which they had made a plentiful provision to "
               "greet the arrival of the young princess. The French, in return, invited the English to a supper, which "
               "was to be given the next day.")
QUESTION = "What animal is the astronaut riding?"
QUESTION_ANSWER = ("The astronaut is riding a **horse** (a chestnut/orange horse, shown in a galloping pose with its mane "
                   "flowing and tail streaming).")
# case -> (system, the user turn's parts: (kind, file or "question": the spoken question), the answer)
QA = {
    "image": (None, [("image", ASTRONAUT), ("text", "Describe the image.")], IMAGE_ANSWER),
    "clip": (A.asr_system("en"), [("audio", SPEECH_CLIP)], CLIP_ANSWER),
    "image+question": (None, [("image", ASTRONAUT), ("audio", "question")], QUESTION_ANSWER),
}


@pytest.fixture(scope="module")
def app() -> tuple[Lfm, object]:
    _files(MODEL, MMPROJ, SPEECH_CLIP)
    m = Lfm(MODEL, mmproj=MMPROJ)
    return m, m.load(capacity=16384)


@pytest.fixture(scope="module")
def question(app: tuple[Lfm, object], tmp_path_factory: pytest.TempPathFactory) -> str:
    """The spoken question: Omni's own speech (seed 1) as a WAV file."""
    m, _ = app
    path = str(tmp_path_factory.mktemp("speech") / "question.wav")
    with open(path, "wb") as f:
        f.write(S.wav(np.concatenate(list(m.tts(QUESTION, seed=1)))))
    return path


def _kl(want: np.ndarray, got: np.ndarray) -> tuple[float, float]:
    lw = want - np.logaddexp.reduce(want.astype(np.float64), -1, keepdims=True)
    lg = got - np.logaddexp.reduce(got.astype(np.float64), -1, keepdims=True)
    kl = (np.exp(lw) * (lw - lg)).sum(-1)
    return float(kl.mean()), float((want.argmax(-1) == got.argmax(-1)).mean())


def _float64_rows(gold: object, media: list[dict], files: list[str]) -> np.ndarray:
    """The media chunks' float64 embeddings (``tests.lfm3vl.ref``, ``tests.lfm3a.ref``) of llama.cpp's inputs: an image's
    pixels as llama.cpp read them (its inp_raw), a clip's samples."""
    rows = []
    for c, f in zip(media, files, strict=True):
        if "image" in c:
            raw = [np.load(os.path.join(gold.path, n)) for n in sorted(os.listdir(gold.path)) if n.startswith("inp_raw.")]
            rows.append(VR.Encoder(MMPROJ).forward(next(r for r in raw if r.ndim == 3 and r.shape[0] == 3)).numpy())
        else:
            rows.append(AR.Encoder(MMPROJ).forward(A.log_mel(A.load_audio(f))).numpy())
    return np.concatenate(rows).astype(np.float32)


@pytest.mark.slow
@pytest.mark.parametrize("case", list(QA))
def test_answer_kl(app: tuple[Lfm, object], question: str, case: str) -> None:
    """Teacher-forced over llama.cpp's answer, the prompt's ids llama.cpp's; the reference: llama.cpp's language model
    (CPU) on the media's float64 embeddings (llama.cpp's own CPU encoders on the Q8_0 mmproj are 3-4% from them). The
    KL with the NPU's encoders within 2x of the NPU language model's own (on the float64 embeddings) or 1.5x of
    llama.cpp's own run's (llama.cpp's level). Measured (own, NPU encoders, llama.cpp's own run): picture 0.0033,
    0.0051, 0.0040; clip 0.0002, 0.0004, 0.0002; picture and spoken question 0.0021, 0.0035, 0.0024."""
    m, npu = app
    system, parts, answer_text = QA[case]
    content, mtmd, files = [], [], []
    for kind, what in parts:
        if kind == "text":
            content.append({"type": "text", "text": what})
            mtmd.append(what)
            continue
        path = question if what == "question" else what
        content.append({"type": kind, kind: path})
        mtmd.append(MARKER)
        files.append(path)
    sys_msg = [{"role": "system", "content": system}] if system else []
    seq = m.chat(sys_msg + [{"role": "user", "content": content}])
    answer = m.encode(answer_text)
    text = _text(m, sys_msg + [{"role": "user", "content": "".join(mtmd)}])
    gold = golden(MODEL, MMPROJ, files, text, answer=answer)
    ids = [t for c in gold.chunks for t in c.get("text", [MEDIA] * (c.get("image", 0) + c.get("audio", 0)))]
    assert [int(t) for t in seq.ids] + answer == ids
    e64 = _float64_rows(gold, [c for c in gold.chunks if "text" not in c], files)
    want = np.asarray(golden(MODEL, MMPROJ, files, text, answer=answer, embd=e64).logits, np.float32)[-len(answer) - 1:-1]
    kls = []
    for s in (Sequence(seq.ids, seq.given, e64, seq.pos, seq.next), seq):
        npu.reset()
        kls.append(_kl(want, np.stack([npu.forward(s)] + [npu.step(t) for t in answer[:-1]])))
    llama = _kl(want, np.asarray(gold.logits, np.float32)[-len(answer) - 1:-1])
    print(f"{case}: KL vs llama.cpp on float64 embeddings: the NPU's language model on them {kls[0][0]:.5f}, with the "
          f"NPU's encoders {kls[1][0]:.5f} (agree {kls[1][1]:.3f}); llama.cpp's own run {llama[0]:.5f}")
    assert kls[1][0] < max(2 * kls[0][0], 1.5 * llama[0]) + 1e-4 and kls[1][1] > 0.9


VIDEO_ASK = "Describe what happens in the video, what you see and hear."
VIDEO_ANSWERS = {  # (llama.cpp's greedy answers, CPU: the soundtrack's narrator is heard, the silent video's not)
    "astronaut": "The video shows a static, digitally rendered scene: a chestnut/orange horse galloping across a sandy "
                 "desert landscape, its front legs lifted in a gallop and kicking up a haze of dust.",
    "astronaut-silent": "The video shows a surreal, digitally rendered scene: a person dressed in a full white "
                        "spacesuit\u2014complete with a reflective helmet, a chest patch, and a long white beard\u2014sits "
                        "astride a glossy chestnut/orange horse that is captured mid-gallop, its front legs lifted and "
                        "tail streaming behind.",
}


def _video_float64(path: str) -> np.ndarray:
    """A video's media chunks' float64 embeddings in llama.cpp's order (per pair its tokens, then its soundtrack rows):
    the pairs' frames as llama.cpp reads them (``tests/test_video.py``: Artlite's, bit for bit), the soundtrack's
    rows split per pair."""
    data = VD.read(path)
    info = VD.probe(data)
    plan = Q.VideoPlan.of(info.n_frames, info.fps, info.width, info.height, heard=True)
    fr = VD.frames(data, info, plan.indices, (plan.width, plan.height))
    track = VD.soundtrack(data, info)
    rows, split = None, None
    if track is not None:
        rows = AR.Encoder(MMPROJ).forward(A.log_mel(track)).numpy()
        split = plan.rows(plan.slices(len(track)), len(rows))
    enc, out = VR.Encoder(MMPROJ), []
    for g, p in enumerate(plan.pairs()):
        x = np.stack([Q.normalized(fr[f]) for f in p])
        out.append(enc.forward(x if len(p) == 2 else x[0]).numpy())
        if split is not None and split[g][1] > split[g][0]:
            out.append(rows[split[g][0]:split[g][1]])
    return np.concatenate(out).astype(np.float32)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(VIDEO_ANSWERS))
def test_video_kl(app: tuple[Lfm, object], videos: str, name: str) -> None:
    """A video with its soundtrack (the astronaut panned, a LibriSpeech narrator) and without (a silent copy),
    teacher-forced over llama.cpp's answer: the prompt's ids llama.cpp's, the KL against llama.cpp's language model on
    the float64 embeddings as ``test_answer_kl``'s. Measured (own, NPU encoders, llama.cpp's own run): with the
    soundtrack 0.0031, 0.0068, 0.0051; silent 0.0034, 0.0065, 0.0048. The pairs carry the encoders' part (the tower's
    tokens 5.1% from float64 at the median, llama.cpp's 3.2%: NPU pairs alone 0.0057), the soundtrack's rows none."""
    m, npu = app
    path = video_file(videos, name)
    seq = m.chat([{"role": "user", "content": [{"type": "video", "video": path}, {"type": "text", "text": VIDEO_ASK}]}])
    answer = m.encode(VIDEO_ANSWERS[name])
    text = _text(m, [{"role": "user", "content": MARKER + VIDEO_ASK}])
    gold = golden(MODEL, MMPROJ, path, text, answer=answer)
    ids = [t for c in gold.chunks for t in c.get("text", [MEDIA] * (c.get("image", 0) + c.get("audio", 0)))]
    assert [int(t) for t in seq.ids] + answer == ids
    e64 = _video_float64(path)
    want = np.asarray(golden(MODEL, MMPROJ, path, text, answer=answer, embd=e64).logits, np.float32)[-len(answer) - 1:-1]
    kls = []
    for s in (Sequence(seq.ids, seq.given, e64, seq.pos, seq.next), seq):
        npu.reset()
        kls.append(_kl(want, np.stack([npu.forward(s)] + [npu.step(t) for t in answer[:-1]])))
    llama = _kl(want, np.asarray(gold.logits, np.float32)[-len(answer) - 1:-1])
    print(f"{name} ({len(seq)} tokens): KL vs llama.cpp on float64 embeddings: the NPU's language model on them "
          f"{kls[0][0]:.5f}, with the NPU's encoders {kls[1][0]:.5f} (agree {kls[1][1]:.3f}); llama.cpp's own run "
          f"{llama[0]:.5f}")
    assert kls[1][0] < max(2 * kls[0][0], 1.5 * llama[0]) + 1e-4 and kls[1][1] > 0.9


@pytest.mark.slow
def test_tts(app: tuple[Lfm, object]) -> None:
    """Omni's speech (its mmproj's sampling: temperature 0.4, no top-k, penalty 1.2 over 64 frames): two seeds end with
    the end of audio, the same seed gives the same samples, and the model's own ASR hears the text."""
    m, npu = app
    text = "Speech synthesis on the NPU sounds clear and natural."
    outs = []
    for seed in (1, 2, 1):
        st = S.SpeechStats()
        y = np.concatenate(list(m.tts(text, seed=seed, stats=st)))
        assert 0 < st.frames < 8 * len(text.split()) and y.size == st.frames * S.FRAME_SAMPLES
        outs.append(y)
    assert np.array_equal(outs[0], outs[2]) and not np.array_equal(outs[0], outs[1])
    norm = lambda t: re.sub(r"[^a-z ]", "", t.lower()).split()
    for y in outs[:2]:
        heard = _asr(m, npu, y)
        print("ASR:", heard)
        assert norm(heard) == norm(text)


def _asr(m: Lfm, npu: object, y: np.ndarray) -> str:
    """The model's own transcript of 24 kHz speech (the offline ASR prompt)."""
    msgs = [{"role": "system", "content": A.asr_system("en")},
            {"role": "user", "content": [{"type": "audio", "audio": A.load_audio(S.wav(y))}]}]
    npu.reset()
    prompt = m.chat(msgs)
    return m.tokenizer.decode([t for t in npu.generate(npu.forward(prompt), 200, m.sampling(), prompt)
                               if t not in m.eos_token_ids])


@pytest.mark.slow
def test_voice_reply_picture(app: tuple[Lfm, object]) -> None:
    """A spoken reply about a picture (interleaved text and speech): the text names the horse, the reply ends with
    ``<|im_end|>`` well before its limits, and the speech says the text (its ASR shares most of its words)."""
    m, npu = app
    st, text, audio = S.SpeechStats(), [], []
    msgs = [{"role": "user", "content": [{"type": "image", "image": ASTRONAUT},
                                         {"type": "text", "text": "What animal is in the picture? Answer in one sentence."}]}]
    for kind, x in m.voice_reply(msgs, seed=3, stats=st):
        (text if kind == "text" else audio).append(x)
    said = m.tokenizer.decode(text)
    print(f"reply: {said!r}; {st.frames} frames")
    assert "horse" in said.lower() and 0 < st.frames < 2048
    heard = _asr(m, npu, np.concatenate(audio))
    print("ASR:", heard)
    norm = lambda t: set(re.sub(r"[^a-z ]", "", t.lower()).split())
    assert len(norm(heard) & norm(said)) >= 0.7 * len(norm(said))


@pytest.mark.slow
def test_realtime_conversation(app: tuple[Lfm, object], question: str) -> None:
    """A realtime conversation on the prompt cache (``models.server.realtime.Conversation`` on an engine over the loaded
    model): two camera frames and a spoken question streamed in 0.1 s pieces (its rows run on the model as they come)
    answered in interleaved speech, then two text questions answered in text; each answer right and the first audio
    soon after the commit. Run again with another request between the turns (the conversation displaced: brought back
    from what the prompt cache kept of it, RAM and a checkpoint, the rest run again): the same sequence and answers.
    The logits there and from scratch as close as one sequence's runs in other splits are (the prefill's 8-bit GEMM
    operands against decode steps: KL ~1e-3; the same split gives the same bits)."""
    from models.lfm3a.speech import SpeechStats
    from models.server.engine import Engine
    from models.server.realtime import Conversation
    m, npu = app
    x = A.load_audio(question)
    example = os.path.join(os.path.dirname(ASTRONAUT), "sd15-example.jpg")  # (another astronaut on another horse)
    other = m.chat([{"role": "user", "content": " ".join(["Tell me about the sea."] * 120)}])

    def session(displace: bool) -> tuple[object, np.ndarray, list[str], float]:
        npu.reset()
        engine = Engine(lambda: (m, npu), warmup=False)
        engine.start()
        assert engine.wait_ready(60)
        run = lambda fn: engine.run(fn).result()
        conv = Conversation(engine, m)

        def respond(mode: str) -> tuple[str, float | None]:
            text, first = [], None
            for kind, y in conv.respond(mode, 200, SpeechStats(), seed=7):
                if kind == "text":
                    text.append(y)
                elif first is None and len(y):
                    first = time.perf_counter()
            return m.tokenizer.decode(text), first

        def elsewhere() -> None:  # (another request: the device runs another sequence)
            if displace:
                run(lambda: (npu.forward(other[engine.hold(other):]), engine.ran(other)))
        try:
            run(lambda: conv.start("Answer in one short sentence.", None))
            run(lambda: conv.image(example))
            run(lambda: conv.image(ASTRONAUT))
            for i in range(0, len(x), 1600):
                run(lambda i=i: conv.audio(x[i:i + 1600]))
            t0 = time.perf_counter()
            run(conv.commit)
            said = [run(lambda: respond("mixed"))]
            elsewhere()
            run(lambda: conv.text("Answer in one word: what colour is the horse?"))
            said.append(run(lambda: respond("text")))
            elsewhere()
            run(lambda: conv.text("How many pictures did I show you?"))
            said.append(run(lambda: respond("text")))
            whole = conv.seq + m.encode("<|im_end|>\n<|im_start|>user\nThanks!")
            logits = np.array(run(lambda: npu.forward(whole[engine.hold(whole):])), np.float32)
            return whole, logits, [t for t, _ in said], said[0][1] - t0
        finally:
            engine.stop()
            engine.join(30)
    whole, cached, said, first = session(False)
    print(f"answers {said}; first audio {first:.2f} s after the commit")
    assert "horse" in said[0].lower() and first < 1.0 and any(w in said[1].lower() for w in ("brown", "chestnut"))
    assert "two" in said[2].lower() or "2" in said[2]
    again, logits, said2, _ = session(True)
    kl, agree = _kl(cached[None], logits[None])
    print(f"displaced between the turns: KL {kl:.2e}")
    assert again.cache_keys() == whole.cache_keys() and said2 == said and kl < 0.05 and agree == 1.0
    npu.reset()
    kl, agree = _kl(np.array(npu.forward(whole), np.float32)[None], cached[None])
    print(f"{len(whole)} positions ({int(whole.given.sum())} rows): KL cached vs from scratch {kl:.2e}")
    assert kl < 0.05 and agree == 1.0  # (the prefill's 8-bit GEMM operands against steps)
