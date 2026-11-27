# MOSS-Transcribe-Diarize on the NPU

MOSS-Transcribe-Diarize 0.9B (architecture `moss`): a Qwen3-0.6B language model behind a Whisper encoder and a VQ
adaptor; a clip's rows take token positions in the prompt as an image's do, and the model answers with timestamps and
speaker tags. It is a Hugging Face checkpoint directory (`-m OpenMOSS-Team/MOSS-Transcribe-Diarize`), not a GGUF file:
the architecture, the vocabulary and the chat template are the checkpoint's own, the weights read from its safetensors
at every load. Running it: [artlite-cli](../cli.md), [artlite-server](../server.md#speech); greedy, as the checkpoint's
`generation_config.json` says (the model has no thinking mode).

```bash
uvx --from huggingface_hub hf download OpenMOSS-Team/MOSS-Transcribe-Diarize \
    --local-dir ~/.artlite/models/OpenMOSS-Team/MOSS-Transcribe-Diarize
artlite-cli -m OpenMOSS-Team/MOSS-Transcribe-Diarize --audio speech.wav
artlite-server -m OpenMOSS-Team/MOSS-Transcribe-Diarize
```

The weights and the code are Apache-2.0 (`OpenMOSS-Team/MOSS-Transcribe-Diarize` on Hugging Face; the GitHub repository
`OpenMOSS/MOSS-Transcribe-Diarize` has the inference code). The transcription instruction is that repository's
`DEFAULT_PROMPT`, verbatim (Apache-2.0); the log-mel is a numpy implementation of `WhisperFeatureExtractor`'s documented
behaviour, not a copy of its code.

## Model files

```
$ARTLITE_MODELS_DIR/OpenMOSS-Team/MOSS-Transcribe-Diarize/
    config.json                    (text_config: Qwen3-0.6B's shapes; audio_config: Whisper's 24 layers)
    generation_config.json          (greedy; eos 151645, max_new_tokens 5,120)
    preprocessor_config.json        (one 30 s chunk: 480,000 samples at 16 kHz, 80 mel bins, n_fft 400, hop 160)
    processor_config.json           (12.5 audio tokens a second, a 4-row merge, a time marker every 5 s)
    tokenizer.json, tokenizer_config.json, special_tokens_map.json, added_tokens.json, merges.txt, chat_template.jinja
    model-00000-of-00001.safetensors   (bf16, 1.82 GB: 683 tensors - the decoder's, the audio tower's, the VQ adaptor's)
```

A checkpoint directory is a model file: every `*.safetensors` in it is read, so the same tensors split into several
files work too (this port's instance keeps the decoder's shard and the tower's apart). The language model's tensors are
`model.language_model.*` (Qwen3's name map) and the tower's `model.whisper_encoder.*` / `model.vq_adaptor.*`, which
the language model's map does not know: the tower reads them by their checkpoint names.

## How the NPU path works

- **The front end** (`models/moss_tr/host.py`): the clip decoded to 16 kHz mono, pad-or-trimmed to one 30 s chunk
  (480,000 samples, the processor's), then the log-mel: 80 bins of the 400-sample frames every 160 under a periodic
  Hann window, the power spectrum, Slaney mel filters, log10 floored 8 dB under the chunk's global maximum, then
  (x + 4) / 4 - 3,000 frames, numpy.
- **The audio span**: the processor's `_compute_audio_token_length`, `(n - 1) // 1,280 + 1` tokens of the clip's own
  samples (1,280 = hop 160 x the encoder's stride 2 x the 4-row merge: 12.5 tokens a second), at most a 30 s chunk's
  375. The encoder runs the whole chunk; a span reads the first `count * 4` of its 1,500 rows. The processor's
  `<|audio_pad|>` becomes the pads plus the time markers' digit tokens (5, 10, ... 30 at 62-token positions, an
  int(12.5 x 5); the span drifts by half a token a marker, as the processor's does): 375 pads and 11 digits = 386 ids
  for a full chunk, in 7 runs of pads.
- **The audio tower** (four programs of the vision overlay, one vision device, each dispatched once at load so that a
  chunk's timing is the steady one): `stem.py`'s conv1 (one dispatch for the chunk, a band of 128 frames at a time: the
  host's im2col panels through a GEMM with the erf-GELU epilogue, 80 -> 1,024 channels), its conv2 (stride 2, one
  dispatch; its A panels assembled from conv1's raw output panels byte for byte, no float round trip), `chain.py`'s 24
  encoder layers as one program dispatched per layer (Whisper's pre-LN, q / k / v of 16 heads of 64 dims, one
  bidirectional softmax over every key of the 1,536-row grid, out_proj, then fc1 / erf-GELU / fc2; the LayerNorm and
  the attention run as code pages: 9,136 B of resident code plus the `ln` page's 7,056 B of the core's 16 KB), then the
  final LayerNorm and the 4-row merge on the host (1,500 rows of 1,024 -> 375 of 4,096) and `adaptor.py`'s VQ adaptor,
  one dispatch: Linear(4,096 -> 1,024) + SiLU + Linear(1,024 -> 1,024) + LayerNorm(eps 1e-6) -> the decoder's
  1,024-wide rows. The position table is added on the host after conv2's GELU (1,500 rows).
- **The decoder** (`models/moss_tr/decoder.py`): the checkpoint's Qwen3-0.6B on the llama decode overlay (28 layers, 16
  query / 8 KV heads of 128, q and k RMS-normed per head, RoPE theta 1e6, `max_position_embeddings` 131,072), its
  output weight the tokens' embedding (the checkpoint has no `lm_head.weight`). The prompt is the chat template's text
  with the tower's rows at the pads: a `Sequence`'s given row per pad (the digits keep their text ids), read through
  the prefill's chunks and the decode steps alike. One hardware context on the array means the tower's dispatch
  invalidates the decoder's overlay, which re-runs its prologue on the next step (`vision_ran`).
- **The transcription** (`Moss.transcribe`): greedy, the official instruction as the user turn's text after the clip
  (the CLI's `--audio` alone calls it, in place of LFM3-Audio's ASR system message; the server's
  `/v1/audio/transcriptions` asks through the chat path with the same instruction as the user turn's text, no guard).
  A turn whose first token is an end-of-sequence - the `[` of a timestamp within a logit of EOS at the
  instruction's near tie - has its first token forced to the best other one; a turn that still ends at once is
  transcribed with the audio alone.

## Speed and accuracy (Strix Halo)

A 30 s chunk, warm (after the load, one process), on the NPU:

| stage | ms |
|---|---|
| log-mel (host) | 12.6 |
| conv1 (one dispatch, the chunk's 3,000 frames) | 12.2 |
| conv2 (one dispatch, its 1,500 rows) | 16.7 |
| the 24 encoder layers (24 dispatches) | 99.9 |
| final LayerNorm, merge and the VQ adaptor (one dispatch) | 39.0 |
| the 414-token prompt through the prefill | 57.8 |
| a decoded token | 11.66 |

| | |
|---|---|
| a 30 s clip, transcribed | 0.0329 / 0.0546 of its 30 s at 64 / 120 tokens (warm, the decoder included) |
| the first chunk after `load` | 0.0374 / 0.0594 |
| a cold process's first chunk | 0.547 / 0.569 |
| `load` | 14.3 s (the tower's build and warm 10.5, the decoder 3.8) |

Accuracy: the decoder's teacher-forced logits against transformers' Qwen3 in float32 on the same 414-token prompt and
8 greedy steps: the peaked positions (the reference's entropy within 0.5 nats: 6 of the 8) keep their argmax and a KL
within 2.1e-4 (the gate: 1e-3); a flat position may drift and flip a near tie (the bf16 activations' tail: the
reference's own bf16 logits carry a 2.9e-4 floor). The tower on a real clip's rows passes the same gate.

Word error rates on LibriSpeech test-clean's 2,620 utterances, greedy, the official instruction: with a padded 30 s
span, 3.83% without the first-token guard and 2.12% with it (24 empty transcripts turned into the right text); with the
clip's own span, 2.29% against the padded span's 3.87% in one session with the guard off in both arms (37 empty turns
fell to 3). That corpus is not in this repository: the tests' clip is test-clean's first chapter
(`tests/datasets/fetch_datasets.py`), and those rates are one-time runs outside it.

## What it does not do

- **One 30 s chunk a prompt.** A longer clip is cut to the chunk and transcribed one chunk at a time, a prompt each,
  the caches reset between them. The upstream processor instead puts every chunk of a recording in one prompt, so its
  chunks see each other's audio; speaker tags and word boundaries across a long recording's chunks can differ from
  the reference's. A clip of up to 30 s is padded to the chunk's 480,000 samples before the log-mel, as the
  processor pads its own chunk.
- **The time markers' interval is the checkpoint's, as a constant.** `host.MARKER_EVERY` is 5 s, the value in this
  checkpoint's `processor_config.json`; a checkpoint with another `time_marker_every_seconds` would need it changed
  (nothing else reads that file).
- **Several clips in one prompt** (`<|audio_pad|>` per clip) work: a superset of the upstream processor, which takes
  exactly one.
- **The answer's budget.** `Moss.transcribe` generates 400 tokens by default (a 30 s transcript's budget; the CLI's
  `-n` overrides it, and the checkpoint's `generation_config.json` allows 5,120), so a transcript that runs past it is
  cut.
- **The first-token guard is Artlite's**, not the model's: upstream forces nothing. It is kept for the near tie above
  and does nothing on a turn whose first token is not EOS.

Design: the module docstrings of `src/models/moss_tr/`.
