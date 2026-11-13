"""LiquidAI LFM3-1.7B on the NPU from a GGUF file (QAD-Q4_0, Q4_K_M, Q8_0; BF16 tensors as BFP16 tiles): 24 layers
of three kinds (short convolutions, sliding-window and full attention), each [operator | SwiGLU FFN] with sandwich
norms (``artlite.model_config.LfmConfig``).

A prompt is token ids or a ``sequence.Sequence`` (token ids and embedding rows: an LFM3-VL image's or video's,
``models.lfm3vl``; an LFM3-Audio clip's, ``models.lfm3a``; the positions 1-D). A dispatch runs up to 4 consecutive tokens through every layer and the LM head on one stream of the
weights
(``models.lfm.overlay.LfmOverlay``): the projections on the main cores and their helpers, the attention, its KV caches
and the conv layers' short convolutions on the operator cores. A prompt runs in chunks of up to 4096 tokens through the
prefill (``models.lfm.prefill.LfmPrefill``: layer-major from one library ELF on the same weight image; the chunk's K / V
land in the decode's caches, its conv rows in the conv layers' histories); its last token's logits come from the LM
head alone. Tokens where a chunk cannot start (inside a cache group of 64 positions) and a prompt's last few run as
decode steps.

The conv histories hold the whole sequence (``recurrent``): a position is kept only where a checkpoint was taken (the
histories, ~0.5 MB). Drafts: the context's (``drafts.Lookup``), whose rejected tokens cost nothing (the histories'
lazy shift, the caches' records overwritten).
"""

import hashlib
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from artlite.decode.design import SLOTS
from artlite.gguf_model import GGUFFile
from artlite.model_config import AudioConfig, ClipConfig, LfmConfig
from artlite.overlay import WeightBuffer
from artlite.vision.device import VisionDevice, load_state, save_state, state_dir
from artlite.vision.weights import sources

from .. import video
from ..common import Model, OverlayModel, Sampling
from ..lfm3a import host as a_host
from ..lfm3a import npu as a_npu
from ..lfm3a import speech as a_speech
from ..lfm3vl import host as vl_host
from ..lfm3vl import npu as vl_npu
from ..sequence import Sequence, splice_media
from ..tokenizer import BpeTokenizer, chat_prompt
from .image import LfmImage
from .overlay import KV_K, LfmOverlay
from .prefill import SIZES, LfmPrefill

# the model card's sampling (the file's general.sampling.* over it)
SAMPLING = {"temp": 0.1, "top_k": 50, "penalty_repeat": 1.1}
# a d1 decision model's images at most (its runner's ``d1_vision_max_pixels``, 1638400: the budget it was trained and is
# served at; its mmproj's maximum is LFM3-VL's)
D1_IMAGE_TOKENS = 1600
KEEP_VIDEOS = 2  # (videos whose sequences are kept)


def media_device(mmproj: str, vision: vl_npu.NpuVision, audio: a_npu.NpuAudio) -> VisionDevice:
    """An mmproj's vision tower and speech networks (LFM3-Omni's: the encoders and the speech generator) built on one
    vision overlay: one hardware context, and an image and a clip run back to back without a switch. Their built state
    is cached together, keyed by the mmproj (``artlite.vision.device.state_dir``)."""
    w = {"vision": vision.weights(), "audio": audio.weights()}
    cache = state_dir("lfm3omni", mmproj, vl_npu.SOURCES + a_npu.SOURCES, [mmproj])
    npu, st = load_state(cache, sources(w), accept=lambda s: vision.accept(s["vision"]) and audio.accept(s["audio"])) \
        or (VisionDevice(), None)
    nets = [vision.build(npu, w["vision"], st and st["vision"], cache), *audio.build(npu, w["audio"], st and st["audio"])]
    if st is None:
        save_state(cache, npu, [n.bd.built() for n in nets], {"vision": vision.state(), "audio": audio.state()})
    npu.warm()
    return npu


class NpuLfm(OverlayModel):
    """The model on the NPU. ``capacity``: the positions the KV caches hold; ``kv_k``: their K precision ("single",
    "hilo"); ``prefill``: prompts in chunks through the prefill (else as decode steps), of the sizes ``sizes``;
    ``weights``: the weight image's buffer, allocated ahead (``WeightBuffer``)."""

    recurrent = True  # (the conv histories hold the whole sequence)
    STEP_COST = (1.0, 1.02, 1.05, 1.06)  # (decode steps at 1K: QAD-Q4_0 20.2 ms for one slot, Q4_K_M 21.3)

    def __init__(self, gguf: GGUFFile, *, capacity: int = 4096, kv_k: str = "single", threads: int = 0,
                 prefill: bool = True, sizes: tuple[int, ...] = SIZES, weights: WeightBuffer | None = None):
        self.gguf = gguf
        self.overlay = LfmOverlay(gguf, max_positions=capacity, kv_k=kv_k, weights=weights)
        self.cfg = self.overlay.cfg
        self.causal = self.cfg.causal  # (d1's decision models: readouts of whole prompts, no decode steps)
        self.eos_token_ids = self.cfg.eos_token_ids
        self.prefill = LfmPrefill(self.overlay, sizes=sizes) if prefill else None
        with ThreadPoolExecutor(1) as pool:
            build = pool.submit(self.prefill.compile) if prefill else None  # (while the weights are packed)
            self.overlay.load(threads=threads)
            for S in range(1, SLOTS + 1) if self.causal else ():  # (the steps' transactions: prompt tokens, a prompt's
                self.overlay.prepare(S, logits=False)  # last, generated ones)
                self.overlay.prepare(S, logits=True)
            if build is not None:
                build.result()  # (its error, if any)
                self.prefill.bind()
        self.overlay.ready()
        self._w_out = gguf.tensor("output_norm.weight").dequantize().reshape(-1)
        self._w_post = gguf.tensor(f"blk.{self.cfg.num_hidden_layers - 1}.post_ffw_norm.weight").dequantize().reshape(-1)
        gguf.release()
        self.position = 0

    @property
    def capacity(self) -> int:
        return self.overlay.max_positions

    def describe(self) -> str:
        ov = self.overlay
        return (f"{ov.image.streamed_bytes() / 1e9:.2f} GB of weights, {ov.sizes['KV'] / 1e6:.0f} MB of KV cache "
                f"({self.capacity} positions, K {'hi + lo' if ov.kp == 2 else 'single'})")

    def reset(self) -> None:
        """Forget the cached positions (the conv histories zero)."""
        self.overlay.reset()
        self.position = 0

    def _rows(self, tokens: list[int] | Sequence) -> np.ndarray:
        """The tokens' embeddings (n, hidden) fp32: their ids' rows, a sequence's given rows."""
        if not isinstance(tokens, Sequence):
            return self.gguf.rows("token_embd.weight", tokens)
        x = np.empty((len(tokens), self.cfg.hidden_size), np.float32)
        t = ~tokens.given
        if t.any():
            x[t] = self.gguf.rows("token_embd.weight", tokens.ids[t])
        if tokens.given.any():
            x[tokens.given] = tokens.rows
        return x

    def audio_step(self, row: np.ndarray, *, logits: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        """An embedding row (hidden,) at the next position (a speech frame's feedback: ``models.lfm3a.speech``), in a
        step without the LM head unless ``logits`` (a text token follows): the hidden state after it (the LM head's
        input, from its header's: ``LfmOverlay.lm_input``: r + v / rms(v) w_post, output-normed; fp32) and the logits
        (vocab,) fp32 or None."""
        if self.position + 1 > self.capacity:
            raise ValueError(f"the KV cache holds {self.capacity} positions")
        out = self._dispatch(lambda: self.overlay.step(np.asarray(row, np.float32).reshape(1, -1), self.position,
                                                       logits=logits))
        self.position += 1
        if logits:
            self.slot = 0
        return self.hidden_after(), None if out is None else self.cap(out[0])

    def hidden_after(self, slot: int = 0) -> np.ndarray:
        """The hidden state (hidden,) fp32 after a slot of the last step, with or without the LM head (module
        ``audio_step``)."""
        v, r, ss = self.overlay.lm_input(slot)
        eps = np.float32(self.cfg.rms_norm_eps)
        u = r + v / np.sqrt(ss / np.float32(self.cfg.hidden_size) + eps) * self._w_post
        return self._normed(u[None])[0]

    def _step_inputs(self, tokens: list[int] | Sequence) -> tuple:
        return (self._rows(tokens),)

    def _chunk_inputs(self, tokens: list[int] | Sequence, *, first: bool) -> tuple:
        return (self._rows(tokens),)

    def _chunk_logits(self, token: int, n: int) -> np.ndarray:
        """(the LM head alone on the chunk's last output row)"""
        x = self.prefill.hidden(n - 1)
        return self._dispatch(lambda: self.overlay.logits_of(x))


class Lfm(Model):
    """An lfm3 GGUF file: the architecture, tokenizer and chat template from its metadata. Thinking: the template opens
    a thought unless ``enable_thinking`` is false (an empty one, ``<think></think>``); a raw text gets no BOS
    (the file's ``add_bos_token``, as llama.cpp). ``mmproj``: the media encoders' file: a vision encoder (images in
    prompts: LFM3-VL's, ``models.lfm3vl.npu.NpuVision``; ``image_min_tokens`` / ``image_max_tokens``: an image's merged
    tokens, default the mmproj's bounds, a d1 decision model's at most ``D1_IMAGE_TOKENS``; videos: ``{"type": "video",
    "video": a file or its bytes}``, ``video_sequence``) and / or a speech encoder (audio clips in prompts: LFM3-Audio's,
    ``models.lfm3a.npu.NpuAudio``; a message's parts ``{"type": "audio", "audio": a file, its bytes or 16 kHz samples}``,
    OpenAI's ``input_audio`` and ``audio_url`` ones); both (LFM3-Omni's mmproj, which also holds the speech generator)
    on one device (``media_device``), a video heard with its soundtrack. Text replies (``sampling``)
    never open speech: the openers ``<|audio_start|>`` / ``<|mixed_start|>`` are banned (``host.TEXT_ONLY_BANNED``:
    transformers' generation mode "text", llama.cpp's logit bias); ``voice_reply`` lets a model that speaks speak."""

    name = "LFM3"
    thinks = True
    reasoning = ("<think>", "</think>")
    thought_first = True
    turn, user_turn = "<|im_start|>", "<|im_start|>user"
    projectors = vl_npu.PROJECTORS + a_npu.PROJECTORS

    def __init__(self, path: str, *, kv_k: str = "single", mmproj: str | None = None, image_min_tokens: int | None = None,
                 image_max_tokens: int | None = None):
        if kv_k not in KV_K:
            raise ValueError(f"kv_k: {kv_k!r} (one of {', '.join(KV_K)})")
        self.gguf = GGUFFile(path)
        self.cfg = LfmConfig.from_gguf(self.gguf)
        self.weights = WeightBuffer(LfmImage(self.gguf, self.cfg))
        self.tokenizer = BpeTokenizer(self.gguf)
        self.eos_token_ids = self.cfg.eos_token_ids
        self.bos = int(self.gguf.meta["tokenizer.ggml.bos_token_id"])
        self.kv_k = kv_k
        self._npu: NpuLfm | None = None
        self._videos: OrderedDict[bytes, Sequence] = OrderedDict()  # (``video_sequence``'s last videos)
        toks = self.gguf.strings("tokenizer.ggml.tokens")
        self.banned = tuple(toks.index(t) for t in a_host.TEXT_ONLY_BANNED if t in toks)
        if mmproj is not None:
            mg = GGUFFile(mmproj)
            has_vision = bool(mg.meta.get("clip.has_vision_encoder", "clip.vision.projector_type" in mg.meta))
            if AudioConfig.has_audio(mg):
                ac = AudioConfig.from_gguf(mg)
                if ac.projector not in a_npu.PROJECTORS:
                    raise NotImplementedError(f"{mmproj}: audio projector {ac.projector!r} (supported: "
                                              f"{', '.join(a_npu.PROJECTORS)})")
                if ac.projection_dim != self.cfg.hidden_size:
                    raise ValueError(f"{mmproj}: embeddings of {ac.projection_dim}, the language model's "
                                     f"{self.cfg.hidden_size}")
                self.audio = a_npu.NpuAudio(mmproj, load=not has_vision)
            elif not has_vision:
                raise ValueError(f"{mmproj}: neither a vision nor a speech encoder")
            if has_vision:
                clip = ClipConfig.from_gguf(mg)
                if clip.projector not in vl_npu.PROJECTORS:
                    raise NotImplementedError(f"{mmproj}: projector {clip.projector!r} (supported: "
                                              f"{', '.join(vl_npu.PROJECTORS)})")
                if clip.projection_dim != self.cfg.hidden_size:
                    raise ValueError(f"{mmproj}: embeddings of {clip.projection_dim}, the language model's "
                                     f"{self.cfg.hidden_size}")
                if image_max_tokens is None and not self.cfg.causal:
                    image_max_tokens = D1_IMAGE_TOKENS
                self.vision = vl_npu.NpuVision(mmproj, min_tokens=image_min_tokens, max_tokens=image_max_tokens,
                                               load=self.audio is None)
            if self.vision is not None and self.audio is not None:
                media_device(mmproj, self.vision, self.audio)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def tts(self, text: str, *, voice: str = "female", language: str = "en", seed: int | None = None,
            max_frames: int = 512, stats: a_speech.SpeechStats | None = None) -> Iterator[np.ndarray]:
        """The text spoken (LFM3-Audio's speech generator: its mmproj; the file's ``tts`` chat template, ``voice``
        female or male, ``language`` a code of ``speech.TTS_LANGUAGES`` or a name): 24 kHz float32 samples in chunks as
        they come (every ``speech.CHUNK`` frames of 80 ms), until the end of audio or ``max_frames``; ``seed``: the speech
        codes' sampling (None: random). The loaded model's positions are the prompt's and the frames'."""
        if self.audio is None or self.audio.gen is None:
            raise ValueError(f"{self.name}: speech output needs LFM3-Audio's mmproj (its speech generator)")
        if self._npu is None:
            raise ValueError(f"{self.name}: load the model first")
        npu = self._npu
        ids = self.encode(a_speech.tts_prompt(self.gguf, text, language=language, voice=voice))
        npu.reset()
        h = npu.outputs(ids, at=[-1])[0]
        sampler = a_speech.CodeSampler(self.audio.gen, seed)
        yield from a_speech.speak(npu, self.audio, h, sampler, max_frames=max_frames, stats=stats)

    def voice_reply(self, messages: list[dict[str, object]], *, mode: str = "mixed", seed: int | None = None,
                    max_tokens: int = 1024, max_frames: int = 2048, reply: a_speech.Reply | None = None,
                    stats: a_speech.SpeechStats | None = None) -> Iterator[tuple[str, object]]:
        """A reply to a conversation that may speak (LFM3-Audio: ``speech.interleave``): ``mode`` "mixed" (interleaved
        text and speech: the system prompt ``speech.INTERLEAVED`` unless the conversation has one, the assistant turn
        opened by ``<|mixed_start|>``), "audio" (``<|audio_start|>``: speech, then text), "auto" (the model switches to
        speech by itself or not), "text" (text only); ("text", token id) and ("audio", 24 kHz samples) as they come;
        ``reply``: filled with the reply's sequence. The loaded model's positions are the conversation's and the
        reply's."""
        if self.audio is None or self.audio.gen is None:
            raise ValueError(f"{self.name}: speech output needs LFM3-Audio's mmproj (its speech generator)")
        if self._npu is None:
            raise ValueError(f"{self.name}: load the model first")
        if mode not in ("mixed", "audio", "auto", "text"):
            raise ValueError(f"mode {mode!r}: mixed, audio, auto or text")
        npu = self._npu
        if mode == "mixed" and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": a_speech.INTERLEAVED}, *messages]
        toks = a_speech.speech_tokens(self.gguf.strings("tokenizer.ggml.tokens"))
        opener = {"mixed": [toks[a_speech.MIXED_START]], "audio": [toks[a_speech.AUDIO_START]]}.get(mode, [])
        prompt = self.chat(messages) + opener
        npu.reset()
        logits, h = None, None
        if mode == "audio":
            h = npu.outputs(prompt, at=[-1])[0]
        else:
            logits = npu.forward(prompt)
        sampler = a_speech.CodeSampler(self.audio.gen, seed)
        yield from a_speech.interleave(npu, self.audio, toks, logits, sampler, max_tokens=max_tokens,
                                       max_frames=max_frames, mode={"auto": "text"}.get(mode, mode), h=h,
                                       speech=mode != "text", reply=reply, stats=stats)

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False,
             variables: dict[str, object] | None = None) -> list[int] | Sequence:
        bos = self.gguf.strings("tokenizer.ggml.tokens", self.bos)[0]
        text = chat_prompt(self.gguf, messages, enable_thinking=think, tools=tools, preserve_thinking=preserve,
                           template=self.chat_template, **{"bos_token": bos, **(variables or {})})
        parts = [p for m in messages if isinstance(m.get("content"), list) for p in m["content"] if isinstance(p, dict)]
        images = [p["image"] for p in parts if p.get("type") == "image"]
        clips = [a_host.part_audio(p) for p in parts if p.get("type") in a_host.AUDIO_PARTS]
        videos = [p["video"] for p in parts if p.get("type") == "video"]
        if not images and not clips and not videos:
            return self.tokenizer.encode(text)
        if (images or videos) and self.vision is None:
            raise ValueError(f"{self.name}: images and videos need its vision encoder (an mmproj file)")
        if clips and self.audio is None:
            raise ValueError(f"{self.name}: audio needs its speech encoder (an mmproj file)")
        try:
            return splice_media(text, {vl_host.IMAGE: [self.vision.sequence(im) for im in images],
                                       a_host.AUDIO: [self.audio.sequence(c) for c in clips],
                                       vl_host.VIDEO: [self.video_sequence(v) for v in videos]}, self.tokenizer.encode)
        finally:
            self._vision_ran()

    def video_sequence(self, src: str | bytes, *, soundtrack: bool = True) -> Sequence:
        """A video (a file or its bytes) as the language model's tokens (``models.lfm3vl.host``: its frames' pairs
        through the vision encoder, ``NpuVision.pairs``; with LFM3-Omni's mmproj, which has the speech encoder too, its
        soundtrack's rows after each pair, unless ``soundtrack`` is false). The last ``KEEP_VIDEOS`` videos' sequences
        are kept (a conversation's video runs the encoders once)."""
        data = video.read(src)
        key = hashlib.sha256(data).digest() + bytes([soundtrack])
        if key in self._videos:
            self._videos.move_to_end(key)
            return self._videos[key]
        info = video.probe(data)
        heard = self.audio is not None  # (the mmproj hears a video: LFM3-Omni's prompt layout)
        plan = vl_host.VideoPlan.of(info.n_frames, info.fps, info.width, info.height, heard=heard)
        frames = video.frames(data, info, plan.indices, (plan.width, plan.height))
        groups = len(plan.pairs(len(frames)))
        rows = None
        if heard and soundtrack and (track := video.soundtrack(data, info)) is not None:
            slices = plan.slices(len(track))[:groups]
            embd = self.audio.embeddings(track[:slices[-1][1]])
            rows = [a_host.clip_sequence(embd[b:e]) for b, e in plan.rows(slices, len(embd))]
        grid = plan.grid
        pairs = [grid.sequence(e) for e in self.vision.pairs(frames)]
        seq = self._videos[key] = vl_host.video_sequence(pairs, plan.stamps(len(frames)), self.tokenizer.encode, rows)
        if len(self._videos) > KEEP_VIDEOS:
            self._videos.popitem(last=False)
        return seq

    def sampling(self, *, think: bool = False) -> Sampling:
        """The file's sampling (general.sampling.*), else the model card's."""
        s = SAMPLING | self.cfg.sampling
        return Sampling(temp=s["temp"], top_k=int(s["top_k"]), top_p=s.get("top_p", 1.0),
                        repeat_penalty=s.get("penalty_repeat", 1.0), repeat_last_n=int(s.get("penalty_last_n", 64)),
                        banned=self.banned)

    def load(self, *, capacity: int, prefill: bool = True) -> NpuLfm:
        self._npu = NpuLfm(self.gguf, capacity=capacity, kv_k=self.kv_k, prefill=prefill, weights=self.weights)
        return self._npu
