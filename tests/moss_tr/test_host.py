"""MOSS-Transcribe's prompt host side (``models.moss_tr.host``): the audio span's ids and markers, the adaptor's rows at
the pads' positions only, and the decode's tied output weight (the checkpoint's embedding)."""

import numpy as np
import pytest

from artlite.hf_model import HfModel
from artlite.model_config import moss_dir
from models.moss_tr.host import (
    AUDIO_PAD,
    CHUNK_FRAMES,
    CHUNK_TOKENS,
    N_MEL,
    N_SAMPLES,
    audio_sequence,
    audio_span,
    fit_chunk,
    layer_norm,
    log_mel,
    mel_filters,
    pad_runs,
    span_tokens,
    splice_audio,
    splice_clips,
)
from models.tokenizer import BpeTokenizer

from ..model_files import require


@pytest.fixture(scope="module")
def tok() -> BpeTokenizer:
    """The checkpoint's tokenizer (the digit tokens' ids are the processor's)."""
    require(moss_dir())
    return BpeTokenizer(HfModel(moss_dir()))


def digits(tok: BpeTokenizer) -> dict[str, int]:
    """The digit tokens' ids, one token each (the processor refuses anything else)."""
    out = {d: tok.encode(d) for d in "0123456789"}
    assert all(len(v) == 1 for v in out.values())
    return {d: int(v[0]) for d, v in out.items()}


def test_audio_span(tok: BpeTokenizer) -> None:
    """375 pads and the 11 marker ids (5, 10 .. 30) = 386 ids in 7 runs; a marker after each 62-pad run (the processor's
    int(12.5 * 5) = 62 pads a 5 s mark) and the 3 pads past the 30 s mark."""
    ids = audio_span(CHUNK_TOKENS, digits=digits(tok))
    assert len(ids) == 386 and int((ids == AUDIO_PAD).sum()) == CHUNK_TOKENS
    runs = pad_runs(ids)
    assert len(runs) == 7 and np.array_equal(runs, np.array([0, 63, 127, 191, 255, 319, 383]))
    assert len(ids) - int(runs[-1]) == 3  # (the last run's pads)
    marks = np.flatnonzero(ids != AUDIO_PAD)
    assert len(marks) == 11
    assert np.array_equal(np.sort(ids[marks]), np.sort([digits(tok)[d] for sec in (5, 10, 15, 20, 25, 30)
                                                       for d in str(sec)]))


def test_span_tokens() -> None:
    """The span's token count is the clip's own (the processor's ``_compute_audio_token_length``: ``(n - 1) // 1,280 + 1``,
    1,280 samples = hop 160 x the encoder's stride 2 x the merge 4): 2 s -> 25, 5 s -> 63, 15 s -> 188, a 30 s chunk ->
    375; a longer clip is cut to the chunk, an empty one keeps one token."""
    for sec, want in ((1, 13), (2, 25), (5, 63), (15, 188), (30, 375)):
        assert span_tokens(sec * 16000) == want
    assert span_tokens(480000 + 16000) == CHUNK_TOKENS and span_tokens(0) == 1 and span_tokens(1) == 1


def test_audio_span_is_the_clips_own(tok: BpeTokenizer) -> None:
    """A clip's span is its count's pads plus one id a marker digit (no marker under 5 s, 11 digit ids at 30 s): 2 s ->
    25 ids, 5 s -> 64, 15 s -> 193, 30 s -> 386, the encoder rows it reads count * 4."""
    d = digits(tok)
    for sec, count, marks in ((2, 25, ()), (5, 63, (5,)), (15, 188, (5, 10, 15)),
                              (30, 375, (5, 10, 15, 20, 25, 30))):
        assert span_tokens(sec * 16000) == count
        ids = audio_span(count, digits=d)
        digits_ = [d[c] for m in marks for c in str(m)]
        assert len(ids) == count + len(digits_) and int((ids == AUDIO_PAD).sum()) == count
        assert np.array_equal(ids[ids != AUDIO_PAD],
                              np.asarray(digits_, np.int64))  # (the markers' digit tokens, in order)


def test_audio_sequence(tok: BpeTokenizer) -> None:
    """One adaptor row per pad (in order), the digits keeping their ids, a position a token."""
    rows = np.arange(375 * 8, dtype=np.float32).reshape(375, 8)  # (8 channels for the test's sake)
    seq = audio_sequence(rows, digits=digits(tok))
    assert len(seq) == 386 and int(seq.given.sum()) == 375
    assert seq.rows.shape == (375, 8) and np.array_equal(seq.rows, rows)
    span = audio_span(375, digits=digits(tok))
    assert np.all(seq.ids[seq.given] == -1)  # (MEDIA: the row stands for the token)
    assert np.array_equal(seq.ids[~seq.given], span[~np.isin(span, AUDIO_PAD)])
    assert np.array_equal(seq.pos[:, 0], np.arange(386))


def test_splice_audio(tok: BpeTokenizer) -> None:
    """A text with one ``<|audio_pad|>``: the span at its place (386 ids), the texts around it tokenized as they are."""
    d = digits(tok)
    rows = np.zeros((375, 1024), np.float32)
    text = "before <|audio_pad|> after"
    seq = splice_audio(text, rows, lambda t: tok.encode_plain(t), digits=d)
    before = tok.encode_plain("before ")
    after = tok.encode_plain(" after")
    assert len(seq) == len(before) + 386 + len(after) and int(seq.given.sum()) == 375
    assert np.array_equal(seq.ids[:len(before)], np.asarray(before))
    assert np.array_equal(seq.ids[-len(after):], np.asarray(after))
    with pytest.raises(ValueError):
        splice_audio("no marker", rows, lambda t: tok.encode_plain(t), digits=d)


def test_tied_head(tok: BpeTokenizer) -> None:
    """The decoder's output weight is the tokens' embedding: the same bytes, one entry (the checkpoint has no
    ``lm_head.weight``)."""
    m = HfModel(moss_dir())
    assert "lm_head.weight" not in m.hf_names()
    assert m.tensor_info["output.weight"] == m.tensor_info["token_embd.weight"]
    assert m.tensor_info["output.weight"][0] == (1024, 151936)
    assert m.hparams["num_hidden_layers"] == 28 and m.hparams["hidden_size"] == 1024


def test_a8_gate() -> None:
    """The A8 criterion for a bf16 decoder: a peaked position's argmax and KL hold (1e-3), a flat one may flip or drift,
    and a peaked position's flip or drift fails."""
    from models.moss_tr.decoder import a8_gate

    vocab, n = 8, 4
    ref = np.full((n, vocab), -20.0, np.float64)  # (a flat tail: 8 tokens share the mass)
    ref[0, 0] = 10.0  # (peaked: one token holds it all)
    ref[1, 0], ref[1, 1] = 5.0, 4.9  # (a near tie, flat)
    ref[2, 0] = 8.0  # (peaked)
    ref[3, :] = 0.0  # (flat: the 8 tokens alike)
    got = np.array(ref, np.float64)
    got[0, 0] += 1e-4  # (a peaked position's bf16-level noise: the KL tiny)
    got[1, 0] = 4.0  # (the near tie flips: allowed, flat)
    got[2, 1] = 9.0  # (a peaked position's argmax lost: fails)
    summary = a8_gate(got, ref)
    assert summary["positions"] == 4 and summary["peaked"] == 2  # (positions 0 and 2)
    assert summary["flips"] == 1 and not summary["ok"]
    drifted = np.array(ref, np.float64)
    drifted[0, 1] = 11.0  # (a peaked position's mass moves to another token: the KL is large)
    assert a8_gate(drifted, ref)["worst_kl_peaked"] > 1e-3
    identical = np.array(ref, np.float64)
    identical[1, 0], identical[1, 1] = 0.0, 0.0  # (a flat position's noise: allowed)
    assert a8_gate(identical, ref)["ok"]


def test_adaptor() -> None:
    """The VQ adaptor's arithmetic against the checkpoint's in float64 (the gate A7): the bf16 input rows, the bfp16
    panels and weights, the command's bf16 biases, SiLU, the second linear and the LayerNorm."""
    from models.moss_tr.adaptor import HID, arithmetic, reference, weights

    rng = np.random.default_rng(5)
    w = {"a0_w": rng.standard_normal((1024, 4096)).astype(np.float32) * 0.02,
         "a0_b": rng.standard_normal(1024).astype(np.float32) * 0.01,
         "a2_w": rng.standard_normal((1024, 1024)).astype(np.float32) * 0.02,
         "a2_b": rng.standard_normal(1024).astype(np.float32) * 0.01,
         "a3_w": 1.0 + rng.standard_normal(1024).astype(np.float32) * 0.1,
         "a3_b": rng.standard_normal(1024).astype(np.float32) * 0.05}
    x = rng.standard_normal((375, 4096)).astype(np.float32) * 0.3
    got = arithmetic(x, weights(w))
    ref = reference(x, weights(w))
    assert got.shape == (375, HID) and np.isfinite(got).all()
    rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    assert rel <= 0.03, f"the adaptor is {rel:.4f} from the reference (seeded weights; 3 %)"


def test_adaptor_program() -> None:
    """The adaptor's push on a fake device (no NPU): one dispatch, its rows' images, the weights' and the two pages it
    runs (``ln`` and ``r``: the resident code and either fit 16 KB)."""
    from artlite.vision.device import VisionDevice
    from models.moss_tr.adaptor import Adaptor

    from ..operators.moss_layer import A_PAGE_BUDGET, page_sizes, program_memory
    from ..txn_dump import fake_vision

    rng = np.random.default_rng(6)
    w = {"a0_w": rng.standard_normal((1024, 4096)).astype(np.float32) * 0.02,
         "a0_b": rng.standard_normal(1024).astype(np.float32) * 0.01,
         "a2_w": rng.standard_normal((1024, 1024)).astype(np.float32) * 0.02,
         "a2_b": rng.standard_normal(1024).astype(np.float32) * 0.01,
         "a3_w": 1.0 + rng.standard_normal(1024).astype(np.float32) * 0.1,
         "a3_b": rng.standard_normal(1024).astype(np.float32) * 0.05}
    with fake_vision() as log:
        npu = VisionDevice()
        ad = Adaptor(npu, w)
        ad.rows_image(np.zeros((375, 4096), np.float32))
        ad.run()
    assert len(log.entries) == 1 and len(ad.program.words) > 0
    sizes = page_sizes(npu.design)
    assert sizes["r"] <= A_PAGE_BUDGET and all(v <= 16384 for v in program_memory(sizes, ("ln", "r")).values())


def test_fit_chunk() -> None:
    """One 30 s chunk: a longer recording cut to 480,000 samples, a shorter one zero-padded, equal lengths unchanged."""
    x = np.arange(N_SAMPLES + 1600, dtype=np.float32)
    cut = fit_chunk(x)
    assert cut.shape == (N_SAMPLES,) and np.array_equal(cut, x[:N_SAMPLES])
    short = fit_chunk(np.ones(16000, np.float32))
    assert short.shape == (N_SAMPLES,) and short[:16000].all() and not short[16000:].any()


def test_mel_filters() -> None:
    """The Slaney bank: 80 filters over the 201 bins of a 400-sample frame (0 .. 8,000 Hz), non-negative, their peaks
    rising with the filter's index (mel scale from low to high)."""
    m = mel_filters()
    assert m.shape == (N_MEL, 201) and m.dtype == np.float32 and (m >= 0).all()
    peaks = m.argmax(1)
    assert (np.diff(peaks) >= 0).all() and peaks[0] < 5 and peaks[-1] > 190


def test_log_mel() -> None:
    """The chunk's log-mel: (80, 3,000), float32, a silent chunk's every cell at the floor's (x + 4) / 4 = -1.5, and a
    chunk with a loud part reaching the floor 8 dB under its own maximum (a span of exactly 2.0)."""
    silent = log_mel(np.zeros(N_SAMPLES, np.float32))
    assert silent.shape == (N_MEL, CHUNK_FRAMES) and silent.dtype == np.float32
    assert np.allclose(silent, -1.5)
    t = np.arange(N_SAMPLES, dtype=np.float32) / np.float32(16000)
    tone = np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32)
    tone[:N_SAMPLES // 2] = 0.0  # (half the chunk silent: the floor is reached)
    m = log_mel(tone)
    assert m.shape == (N_MEL, CHUNK_FRAMES) and abs(float(m.max() - m.min()) - 2.0) < 1e-5
    assert np.isfinite(m).all()


def test_layer_norm() -> None:
    """The final LayerNorm (float64 statistics, gamma / beta): a row's mean 0 and variance 1 before gamma."""
    x = np.random.default_rng(7).standard_normal((4, 16)).astype(np.float32) * 3.0 + 5.0
    w, b = np.ones(16, np.float32), np.zeros(16, np.float32)
    y = layer_norm(x, w, b, 1e-5)
    assert np.allclose(y.mean(1), 0, atol=1e-6) and np.allclose(y.var(1), 1, rtol=1e-3)


def test_splice_clips(tok: BpeTokenizer) -> None:
    """Two clips at a text's two ``<|audio_pad|>`` places: their spans in order, the texts around them as they are."""
    d = digits(tok)
    rows = [np.zeros((375, 1024), np.float32), np.ones((375, 1024), np.float32)]
    clips = [audio_sequence(r, digits=d) for r in rows]
    text = "a <|audio_pad|> b <|audio_pad|> c"
    seq = splice_clips(text, clips, tok.encode_plain)
    assert int(seq.given.sum()) == 750 and len(seq) == len(tok.encode_plain("a ")) + 2 * 386 + len(tok.encode_plain(" b ")) \
        + len(tok.encode_plain(" c"))
    assert seq.rows.shape == (750, 1024) and np.array_equal(seq.rows[:375], rows[0])
    with pytest.raises(ValueError):
        splice_clips("one marker", clips, tok.encode_plain)
