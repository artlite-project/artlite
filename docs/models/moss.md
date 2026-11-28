# MOSS-Transcribe-Diarize on the NPU

OpenMOSS's MOSS-Transcribe-Diarize 0.9B (Apache-2.0) transcribes a recording with timestamps and speaker tags: a
Whisper encoder and a VQ adaptor (its audio tower) feed a Qwen3-0.6B language model, which writes
`[start][S01]text[end]` segments, times in seconds from the recording's start, anonymous speakers. Running it:
[artlite-cli](../cli.md) (`--audio` alone: a transcript per file), [artlite-server](../server.md#speech)
(`/v1/audio/transcriptions`, `response_format` `diarized_json` for the segments).

```bash
uvx --from huggingface_hub hf download OpenMOSS-Team/MOSS-Transcribe-Diarize \
    --local-dir ~/.artlite/models/OpenMOSS-Team/MOSS-Transcribe-Diarize
artlite-cli -m OpenMOSS-Team/MOSS-Transcribe-Diarize --audio meeting.wav
artlite-server -m OpenMOSS-Team/MOSS-Transcribe-Diarize
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@meeting.wav -F response_format=diarized_json
```

## Model files

```
$ARTLITE_MODELS_DIR/
  OpenMOSS-Team/MOSS-Transcribe-Diarize/              (the checkpoint directory: BF16, 1.8 GB)
  handy-computer/moss-transcribe-diarize-gguf/
      MOSS-Transcribe-Diarize-Q4_K_M.gguf             (transcribe.cpp's files: Q4_K_M, Q5_K_M, Q8_0, BF16, ...)
```

Either is the whole model in one place: the checkpoint directory opens as llama.cpp-style files (`artlite.hf_model`:
its language model as architecture `moss`, its tower as an mmproj of gguf-py's Whisper names), a
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) GGUF file likewise (`artlite.transcribe_cpp`).
A quantized file's language model reads its own tiles (Q4_K_M: Q4_K, Q8_0 and a screened Q6_K head, 0.34 GB a token
against the checkpoint's 0.59); the tower takes 9-bit operands either way (a K-quant matrix dequantized at load).

## How it runs

- **The recording** (`models/moss/host.py`, the checkpoint's processor): 16 kHz mono (`models.audio.load_audio`: WAV,
  MP3, FLAC, Ogg Vorbis, any rate), cut into chunks of 30 s, the last zero-padded; each chunk's log-mel spectrogram (80
  bins, 3000 frames; Whisper's, numpy on the host). A chunk gives a token per 80 ms of its own samples (375 for a full
  chunk), and the prompt takes all of the recording's tokens at once, the elapsed seconds as digit tokens every 5 s
  (the checkpoint's `processor_config.json`), as the model was trained: a recording of any length is one prompt and
  its speakers keep their tags through it (the model card: up to 90 minutes, 762 positions a minute).
- **The tower** (`models/moss/encoder.py`) runs on the vision overlay, a dispatch per chunk (~88 ms): conv1 and conv2
  as GEMMs of the host's panels (conv1 at the three frames each conv2 row reads), the position rows, 24 layers of
  attention over the chunk's 1500 rows (a token's 4 rows in one window of the overlay's window layout, so that the
  4-row merge is its), the final norm and the adaptor, written as the language model's rows. Its weights are packed
  from the model file at every load, its program cached (a load 0.4 s).
- **The language model** is Qwen3-0.6B on Llama's path ([Qwen3](qwen3.md): its decode overlay and prefill), the tied
  output weight its head, the tower's rows in place of the audio tokens' embeddings. Greedy (the checkpoint's
  `generation_config.json`), no thinking mode. A transcript's form gives draft tokens: after an end time, the next
  segment's start time and speaker tag are mostly the end time's and the last tag (`host.Drafts`), checked 4 at a step
  with the context's (+30% tokens per second on a 30 s clip).
- **The transcription prompt** is the model card's instruction (Chinese; it asks for the timestamps and the speaker
  tags) after the recording; artlite-server's `prompt` becomes its hotword hint (`热词提示：`, the card's recipe).
  The model finds the language itself (`language` is not used).

## Speed and accuracy (Strix Halo)

| a transcription of | checkpoint (BF16) | transcribe.cpp Q4_K_M |
|---|---|---|
| 3.5 s, 24 tokens | 0.42 s | |
| 30 s, 224 tokens | 2.3 s: the tower 0.10, the prompt 0.05, the text 2.11 (106 tokens/s) | 1.5 s (the text 1.39: 162 tokens/s) |
| 89 s of three speakers, 527 tokens | 6.8 s | 4.9 s |
| 10 minutes, 4480 tokens (one prompt of 7891) | 103 s: the tower 1.7, the prompt 1.5, the text 100 (45 tokens/s); `--kv-k single` 83 s | |
| LibriSpeech test-clean (2620 utterances, 5.4 h) | 27 min (one utterance at a time) | |

Decode is the reading of the weights (a step of one token: 12 ms with the checkpoint's 0.59 GB, 7.8 ms with Q4_K_M's
0.34) and, deep in a long recording, of the KV cache (96 KB a position, K as hi + lo bfp16 blocks: at 10 minutes a step
takes 30 ms; `--kv-k single`, one block per 8 dims: 64 KB, 24 ms, a WER of 2.12% instead of 1.99% on test-clean, 5
empty transcripts instead of the official model's 3). For
comparison, transcribe.cpp on an M4 Max's GPU (Metal) takes 1.4 s for a 35 s clip, on a Ryzen 7 4750U's iGPU 11 s.

Word error rate on LibriSpeech test-clean (all 2620 utterances, greedy, Whisper's English normalizer, the segments'
markers dropped): **1.99%** (95% interval 1.74 - 2.31%), where the author's repository (bf16) gives 2.07% and
transcribe.cpp's BF16 / Q8_0 / Q4_K_M files 2.08 / 1.93 / 2.59% (their model card); 3 utterances come out empty, as
the official model's own (transformers, fp32) do. The 10-minute recording (three chapters of two speakers' reading):
0.60%, a new speaker tag at each chapter's start.

Against the checkpoint computed exactly (float64 tower, fp32 language model), teacher-forced over a transcript: KL
0.0007 (the NPU's tower and prefill; with the exact tower's rows 0.0005), the same argmax at every token. The tower's
rows lie 1.5% (median) from the exact ones; the model's own bf16 is 0.7% off. A few rows lie further (up to a third
of their norm): the encoder's layer 7 writes a massive activation (~2500 against ~2) into a few rows, and a row that
gets a little of it gets as much as the last bits of its input decide (torch bf16: up to 8%). The transcripts are the
official model's word for word; timestamps may differ by 0.01 - 0.06 s.

NPU code: `src/models/moss/` (the model object `Moss`: `moss.py`).
