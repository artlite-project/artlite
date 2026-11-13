# LFM3 on the NPU

LiquidAI's LFM3-1.7B (and LFM3-VL-2B with images and videos, LFM3-Audio-2B with speech, LFM3-Omni-2B with all of them,
the decision models d1-2B and d1-Omni-2B: [below](#lfm3-vl-2b-images)) from a GGUF file (architecture `lfm3`):
24 layers of three kinds, 13 short-convolution layers, 6
full-attention layers and 5 sliding-window attention layers (a window of 256 positions), each an operator and a SwiGLU
FFN with norms before and after both. The architecture is read from the file's metadata, the tokenizer and the chat
template from the file itself. Running it: [artlite-cli](../cli.md); default sampling: the file's (temperature 0.1,
top-k 50, repetition penalty 1.1).

Thinking: `artlite-cli` gives the model an empty thought unless `--think`; `artlite-server` lets it think when a request
asks (`-rea on`, a request's `chat_template_kwargs.enable_thinking`), and keeps the earlier answers' thoughts
(`--reasoning-preserve`, the template's `preserve_thinking`). Tool calls: the template's Qwen XML calls, deferred tools
by the template ([artlite-server](../server.md)).

## Model files

```
$ARTLITE_MODELS_DIR/LiquidAI/LFM3-1.7B-GGUF/     LiquidAI/LFM3-1.7B-GGUF
    LFM3-1.7B-QAD-Q4_0.gguf                      (Liquid's quantization-aware-distilled 4-bit: the nearest to the BF16 model)
    LFM3-1.7B-Q4_K_M.gguf
    LFM3-1.7B-Q8_0.gguf
    LFM3-1.7B-BF16.gguf                          (the fp32 reference only, for now: tests/lfm/cpu.py)
```

The models directory: [Installation](../install.md#models-directory).

## How the NPU path works

Paths below are under `src/models/` (LFM3's own code: `lfm/`) and `src/artlite/` (what the models share). One design,
compiled once: the decode overlay shared with Llama, Gemma and Qwen3.5 (`artlite.decode`). The host builds every
dispatch's transaction (`lfm/overlay.py`, `LfmOverlay`); the cores read what to do from the metadata of their input
objects.

- A dispatch runs up to 4 token slots through all 24 layers and the LM head, streaming the weights once for all of
  them.
- Cores per column: the main core and its helper run the projections; the norms after each operator and FFN
  update the residual stream in fp32 (Gemma 4's post-norm form). The operator core runs the attention of its half of a
  KV head (two columns per KV head: each its two query heads and half of the head's K and V; a sigmoid output gate per
  head; the full layers rotate 32 of the 64 dimension pairs, the sliding-window layers all of them) and the conv
  layers' short convolution (`lfm/conv.py`: B x into each channel's history of 6 rows, y = C conv(B x)). Its code
  (two attention kinds, the gate, the conv) is larger than a core's 16 KB of program memory: three pages, loaded from the
  memory tile when a step needs them (`artlite.operators.paging`).
- Drafts from the context (`models/drafts.py`) cost nothing when rejected: a step's conv rows are committed by the next
  step's position, its cache records are overwritten.
- The prompt runs on a prefill (`lfm/prefill.py`, `LfmPrefill`): chunks of up to 4096 tokens through all 24 layers, as
  GEMMs on the same weights (`prefill/gemm.py`), the conv (`lfm/conv_prefill.py`, one core per column), the attention
  shared with Llama and Qwen3 (`prefill/attn.py`: one cache stream per KV head for its two columns, the gates, the
  sliding windows) and fp32 norms (`prefill/norm.py`). It writes the KV caches and the conv histories the decode
  continues from; the prompt's last token's logits come from the LM head alone.
- The conv histories hold the whole sequence: a position is kept only where a checkpoint was taken (`artlite-server`'s
  prompt cache: a checkpoint is 2.8 MB, copied in under a millisecond).

Memory:
- The weights the NPU streams (QAD-Q4_0: 1.07 GB; Q4_K_M: 1.12 GB; the LM head is the token embedding, Q6_K) are
  re-laid-out into a device buffer at load, in memory, at their quantization's size. Nothing is written to disk.
- The KV cache: 108 MB at 8192 positions, 1.68 GB at 131072 (K as one bfp16 block per 8 dimensions; `--kv-k hilo`: hi
  + lo blocks). The sliding-window layers' caches are linear in the positions too (the prefill writes and reads them
  as the decode does).

Speed and accuracy (Strix Halo):

| | QAD-Q4_0 | Q4_K_M |
|---|---|---|
| time to first token, 1K / 8K / 32K / 64K tokens | 0.153 / 1.19 / 5.93 / 15.0 s | 0.174 / 1.35 / 6.58 / 16.3 s |
| prefill, 1K / 32K tokens | 6674 / 5525 tokens/s | 5896 / 4981 tokens/s |
| generation after 1K / 32K tokens | 50.0 / 41.5 tokens/s | 47.6 / 39.8 tokens/s |
| a step of 2 / 3 / 4 tokens against one | +2% / +5% / +6% (20.2 ms for one at 1K) | |
| teacher-forced KL against the fp32 evaluation (a chat / an essay; llama.cpp's own on the same file) | 5.7e-4 / 5.5e-4 (7.0e-4 / 6.3e-4) | 1.1e-3 / 8.7e-4 (5.1e-3 / 2.5e-3) |
| the same after a prompt through the prefill | 1.1e-3 / 7.3e-4 (the prefill's 8-bit GEMM operands) | 1.3e-3 / 1.0e-3 |
| tool calls: BFCL v3 simple / multiple / parallel, 50 each, thinking off | 140 / 150 (llama-server on the same file: 140) | |
| GSM8K, the first 150 test questions, greedy, thinking off | 78.0% (llama.cpp on the CPU, the same requests: 80.7%; 7 questions right on the NPU alone, 11 on the CPU alone) | |

Q8_0: KL 6.2e-4 / 5.7e-4 (llama.cpp 1.0e-3 / 5.2e-4). Steps of 1-4 tokens give the same logits bit for bit, and a greedy
text with the context's drafts is the text without them.

Against FastFlowLM 1.0.12 on the same NPU (its file of the same QAD Q4_0 weights, at most 32K positions): the first token
8.4x sooner at 1K tokens (1.28 s there), 4.8x at 32K (28.2 s); generation 1.2x / 1.8x as fast after 1K / 32K (41.1 / 23.6
tokens/s). Its documentation reports 81.3% on the first 150 GSM8K questions (its own requests) against its CPU engine's
87.3%, the difference its NPU attention's.

Against llama.cpp's Vulkan backend on the iGPU (the same QAD-Q4_0 file): a prompt runs 1.16x as fast at 1K tokens and
1.9x at 32K, with ~3.3x less energy; generation runs at ~30% of its speed (168.8 / 122.5 tokens/s after 1K / 32K):
[Where the NPU fits](../npu-vs-igpu.md#a-small-hybrid-models-prompts-lfm3-17b).

## LFM3-VL-2B (images)

```bash
artlite-cli -m LiquidAI/LFM3-VL-2B-GGUF/LFM3-VL-2B-QAD-Q4_0.gguf \
    --mmproj LiquidAI/LFM3-VL-2B-GGUF/mmproj-LFM3-VL-2B-BF16.gguf --image photo.jpg -p "Describe the image."
```

```
$ARTLITE_MODELS_DIR/LiquidAI/LFM3-VL-2B-GGUF/    LiquidAI/LFM3-VL-2B-GGUF
    LFM3-VL-2B-QAD-Q4_0.gguf                     (its language model: LFM3-1.7B's shapes, its own weights)
    mmproj-LFM3-VL-2B-BF16.gguf                  (its vision encoder; Q8_0 too)
```

LiquidAI's LFM3-VL-2B is LFM3's trunk with a vision encoder: the language model runs as LFM3-1.7B does (above), its
image tokens take the encoder's rows in place of token embeddings (one position each). `--mmproj`
(artlite-cli and [artlite-server](../server.md), whose chat messages then take `image_url` parts) loads the
encoder. An image is resized whole, as transformers' processor and llama.cpp resize it: its sides to multiples of 32
pixels at its aspect, between `--image-min-tokens` and `--image-max-tokens` tokens of 32 x 32 pixels (defaults 64 and
4096, llama.cpp's; a 1024 x 768 photo is 768 tokens).

The encoder (`lfm3vl/encoder.py`) runs on the vision overlay (in a hardware context of its own): 14 layers over the
image's 16 x 16 patches, 8 of them a gated depthwise 3 x 3 convolution over the patch grid (its own op on the cores:
each patch's 8 neighbours read in strips of their 2 x 2 windows, `vision/windows.py`) and 6 attention over all of
the image's patches (16 query heads sharing 8 key / value heads, a 2-D RoPE), each with a SwiGLU feed-forward; then
each 2 x 2 window merged into one token by an MLP.

Its time grows with the patches and, through the attention, with their square:

| image (tokens) | 512 x 512 (256) | 1024 x 1024 (1024) | 2048 x 2048 (4096) |
|---|---|---|---|
| Artlite (NPU) | 0.036 s | 0.16 s | 1.19 s |
| llama.cpp, iGPU (Vulkan) | 0.14 s | 0.47 s | 3.0 s |
| llama.cpp, CPU (16 threads) | 0.27 s | 1.1 s | 7.8 s |

(Strix Halo, the BF16 mmproj; llama.cpp's `mtmd batch encoding` time, the median of fresh runs.) A large image runs as a
dispatch per one or a few layers (each well under the driver's 2 s). Its weights are converted to bfp16 at every load
(the patch embedding's and the merger's MLP's as bfp16 hi + lo pairs; the pixels enter as their exact 8-bit values), the
dispatches of each image size cached in the cache directory; an image's embeddings are kept, so a chat's next turns do
not run it again.

The time to the first token of an image prompt (the encoder, then the prefill), through each runtime's server
(`image_ttft.py` of the README's image chart: the image and "Describe this image in detail.", QAD-Q4_0 and the BF16
mmproj; the iGPU's best of 3 sessions):

| image (its size: prompt tokens) | Artlite (NPU) | llama-server, iGPU (Vulkan) |
|---|---|---|
| sd15-astronaut.jpg (512 x 512: 282) | 0.145 s | 0.189 s |
| test-1.jpeg (640 x 488: 324) | 0.169 s | 0.251 s |
| hk_OCR.jpg (1300 x 858: 1132) | 0.444 s | 0.914 s |
| airplane.jpeg (1920 x 1440: 2724) | 1.14 s | 2.40 s |

The embeddings are ~3.5% per token from an exact (float64) encoder (the bfp16 operands of its GEMMs), llama.cpp's ~1%;
a few tokens are further off (up to ~0.8 of their norm): patches whose values nearly cancel before the final LayerNorm,
which then scales up their small errors. The language model hardly notices: teacher-forced over a description of an image, its answer stays as close to llama.cpp's own
(KL 0.0007, argmax agreement 1.0) as with llama.cpp's own image embeddings (KL 0.0005).

### Videos

```bash
artlite-cli -m LiquidAI/LFM3-VL-2B-GGUF/LFM3-VL-2B-QAD-Q4_0.gguf \
    --mmproj LiquidAI/LFM3-VL-2B-GGUF/mmproj-LFM3-VL-2B-BF16.gguf --video clip.mp4 -p "What happens in the video?"
```

A video (`--video`, artlite-server's `input_video` / `video_url` parts) needs FFmpeg's `ffmpeg` and `ffprobe` on PATH,
which read its frames and its soundtrack with llama.cpp's commands, so that a prompt is llama-server's for the same
file (its frames bit for bit, its token ids). As transformers' and llama.cpp's processors sample a video: 2 frames a
second, at least 4 and at most 768 (LFM3-Omni-2B, which hears the soundtrack: 384), all at one size whose pixels over
every frame stay under ~25 Mpx (a 30 s 720p video's frames are 832 x 480); consecutive frames in pairs, each pair one
"image" whose two frames the encoder's patch embedding takes together, after a timestamp (`<0.5 seconds>`). LFM3-Omni
hears the soundtrack too: it is encoded as one clip and its rows go after the pair they belong to. A pair runs on the
vision overlay as an image of its size does (a 640 x 360 video's pairs: 35 ms each; 960 x 540: 70 ms), two in one run
where that takes fewer bands (pairs of up to 64 tokens, a video of over ~3 minutes: a band of 128 windows for two; each
pair's embeddings bit for bit its own run's); a frame over the encoder's largest image (4096 tokens: a 4K video under
3 s) takes an image's size, which llama.cpp does not cap. The pairs encode while FFmpeg decodes the video, and a video
whose frames keep to its frame rate (the usual file) has only its sampled frames converted (the same pixels as
llama.cpp's command, which converts every frame).

## LFM3-Audio-2B (speech)

```bash
M=LiquidAI/LFM3-Audio-2B-GGUF
artlite-cli -m $M/LFM3-Audio-2B-Q4_K_M.gguf --mmproj $M/mmproj-LFM3-Audio-2B-Q8_0.gguf --audio speech.wav
artlite-cli -m $M/LFM3-Audio-2B-Q4_K_M.gguf --mmproj $M/mmproj-LFM3-Audio-2B-Q8_0.gguf --audio speech.wav --live realtime
artlite-cli -m $M/LFM3-Audio-2B-Q4_K_M.gguf --mmproj $M/mmproj-LFM3-Audio-2B-Q8_0.gguf --tts "Hello there." -o hello.wav
artlite-cli -m $M/LFM3-Audio-2B-Q4_K_M.gguf --mmproj $M/mmproj-LFM3-Audio-2B-Q8_0.gguf --audio question.wav --speak reply.wav
artlite-server -m $M/LFM3-Audio-2B-Q4_K_M.gguf --mmproj $M/mmproj-LFM3-Audio-2B-Q8_0.gguf
```

```
$ARTLITE_MODELS_DIR/LiquidAI/LFM3-Audio-2B-GGUF/   LiquidAI/LFM3-Audio-2B-GGUF
    LFM3-Audio-2B-Q4_K_M.gguf                      (its language model: LFM3-1.7B's shapes, its own weights; Q4_0 too)
    mmproj-LFM3-Audio-2B-Q8_0.gguf                 (its speech encoder and its speech generator)
```

LiquidAI's LFM3-Audio-2B is LFM3's trunk with a speech encoder: the language model runs as LFM3-1.7B does (above), a
clip's rows (one per 80 ms of audio) take token positions in place of token embeddings. `--mmproj` (artlite-cli and
[artlite-server](../server.md#speech)) loads the encoder. Audio files: WAV, read natively; MP3, FLAC and Ogg Vorbis
through the `miniaudio` package (llama.cpp decodes with the same library), as 16 kHz mono.

- **Transcription**: `--audio` without a prompt (artlite-server: `/v1/audio/transcriptions`) is the model card's ASR
  prompt, the system message "Perform ASR in English." (`--language`; without it "Perform ASR.", the model tells the
  language), greedy, text only (the tokens that would start speech output are never sampled). A clip of any length is
  one prompt, as in llama.cpp, but the model was trained on clips of up to 30 s: past ~3 minutes it starts to repeat
  itself (llama.cpp's does too); the server bounds a transcription's tokens by the audio's length.
- **Live transcription** (English; `--live`, artlite-server's `/v1/realtime?intent=transcription`): the model card's
  live layout, the text of every 0.96 s of audio right after it (12 rows, then the model's text up to its end-of-chunk
  token), restarting on a new line after at most 30 s, so any length works.
- With a prompt, the audio goes into the user message as images do: questions about a clip.

The encoder (`lfm3a/encoder.py`) runs on the vision overlay, in a hardware context of its own: the clip's log-mel
spectrogram (128 bins, 100 frames a second, numpy on the host) through a downsampler of three stride-2 convolutions
(a row per 8 frames), 17 layers (10 short convolutions, 7 attention layers over a causal window of 32 rows) and the
projector, in bands of 512 rows (41 s of audio), one dispatch a band (~18 ms), the layers' state carried from band to
band. Live, the band in progress runs again as each 0.96 s arrives. Its rows are ~2.3% from an exact (float64)
encoder (llama.cpp's ~2.4%), and the language model teacher-forced over a transcript stays as close to llama.cpp's
(KL 0.0002) as with llama.cpp's own rows.

Word error rates on LibriSpeech (200 utterances each of test-clean and test-other, Whisper's English normalizer),
through each runtime's server:

| | test-clean | test-other |
|---|---|---|
| Artlite (NPU), Q4_K_M | 2.09% | 4.51% |
| Artlite (NPU), Q4_0 | 2.11% | 4.87% |
| llama.cpp, CPU / iGPU (Vulkan), Q4_K_M | 2.14% / 2.17% | 4.51% / 4.46% |
| live: Artlite (NPU) | 3.00% | 7.50% |
| live: llama.cpp (Vulkan) | 3.02% | 7.45% |
| whisper.cpp, Whisper base.en | 4.00% | 8.97% |
| FastFlowLM 1.0.9 (NPU), Whisper large-v3-turbo | 4.98% | 7.53% |

A transcription's time, and its energy (the SoC's power above a ~18 W idle), through each runtime's server (Strix
Halo, HTTP included, warm):

| a transcription of | 30 s of speech | 179 s (one prompt) |
|---|---|---|
| Artlite (NPU) | 2.2 s, 21 J | 13.0 s, 120 J |
| llama.cpp, CPU (16 threads) | 1.9 s, 73 J | 10.6 s, 362 J |
| llama.cpp, iGPU (Vulkan) | 1.0 s, 32 J | 6.2 s, 189 J |
| whisper.cpp base.en, CPU / iGPU (Vulkan) | 0.63 s, 18 J / 0.33 s, 7 J | 2.9 s, 90 J / 1.5 s, 40 J |
| FastFlowLM 1.0.9 (NPU), Whisper large-v3-turbo | 6.1 s, 54 J | 26 s, 200 J |

Most of a transcription is the language model writing the text, a decode step per token (~21 ms on the NPU, the
reading of its weights; the encoder takes ~18 ms per 41 s of audio, the prompt ~0.1 s): the iGPU writes faster, the NPU
with two thirds of its energy (a third of the CPU's) and leaves the CPU and the iGPU free.

Live (the 179-s chapter paced as a microphone's, through artlite-server's websocket; `artlite-cli --live realtime`
alike): each 0.96 s of audio takes the NPU ~170 ms (the encoder's band ~27 ms, then its 12 rows and its text as decode
steps), busy 17% of the audio's time, with 4% of a CPU core; a chunk's text comes ~0.18 s after its audio (llama.cpp's
live mode on the iGPU: ~0.05 s). As fast as it goes, the 179 s take 32 s.

### Speech output

The mmproj's speech generator turns the language model's hidden states into speech, as llama.cpp's `llama-tts` and
llama-server do:

- **Text to speech** (`--tts`, artlite-server's `/tts` and `/v1/audio/speech`): the file's tts template, "Perform TTS
  in American English using a female voice." (`--language`: ar, da, de, el, en, en-gb, es, fi, fr, he, it, ja, ko, nl,
  pl, pt, sv, tr, zh, or a language's name; `--voice female` or `male`; the model has no voice cloning), until the
  model's end of audio (or `-n` frames of 80 ms, 512 by default), 24 kHz, streamed every 0.32 s of speech. One
  prompt whatever the text's length, as llama.cpp's.
- **Spoken replies** (`--speak`, a chat's `modalities: ["text", "audio"]`, artlite-server's realtime conversations):
  the model's interleaved text and speech (blocks of 6 text tokens and 12 frames; after `<|text_end|>` speech until its
  end, then text), the text printed as it comes; a question can be audio (`--audio`).

A frame (80 ms of speech) is a decode step of the language model without its LM head (the hidden state read from the
LM head's input), then the mmproj's depthformer on the vision overlay: 12 codebooks in 3 passes of 4 (a frame's own
random order; 3 layers over the 12 codebooks' slots, each pass's logits in fp32), the codes sampled on the host with the
model card's settings (temperature 1, top-k 128, a repetition penalty of 1.1), their 12 embedding rows summed into the
next step's input. Every 4 frames the codec decoder (8 layers over 6 positions a frame: short convolutions, attention
over a window of 30) on the vision overlay turns them into a log-magnitude and phase spectrum, the host its samples
(inverse FFT, overlap-add). Each frame switches the NPU between the language model's hardware context and the vision
overlay's twice (~2.5 ms each).

The depthformer's logits are within KL ~8e-4 per codebook of an exact (float64) depthformer (llama.cpp's: ~4e-4; the
bfp16 weights and inputs), the codec's samples 26.7 dB from an exact codec's (a bfloat16 codec, the reference's dtype:
28.9 dB), and none of it shows in the speech: 200 LibriSpeech sentences spoken twice each and transcribed back by
Whisper base.en have a word error rate of 3.32%, llama.cpp's TTS (CPU) 3.47% (paired: -0.15 points, 95% interval
-0.49 .. +0.16); neither ran past 512 frames.

Text to speech through each runtime's server (`/tts`, streamed; Strix Halo, warm; energy: the SoC's power above a
~18 W idle):

| | 9 s of speech | 41 s of speech | first audio | energy per minute of speech |
|---|---|---|---|---|
| Artlite (NPU), Q4_K_M | 3.5 s | 16.4 s | 0.17 s | 390 J |
| Artlite (NPU), Q4_0 | 3.6 s | 15.8 s | 0.17 s | 420 J |
| llama.cpp, CPU (16 threads), Q4_K_M | 2.3 s | 10.4 s | 0.2 s | 790 J |
| llama.cpp, iGPU (Vulkan), Q4_K_M | 1.6 s | 7.6 s | 0.11 s | 470 J |

A frame takes the NPU ~31 ms (the language model's step ~17 ms, the depthformer's 4 dispatches ~7 ms, the two
switches between the hardware contexts ~5 ms, the codec ~0.7 ms), 2.5x real time; the iGPU is faster, the NPU takes
half of the CPU's energy and somewhat less than the iGPU's, and leaves both free (its host side still keeps the CPU
cores at ~10 W).

## LFM3-Omni-2B (images, videos, speech)

```bash
M=LiquidAI/LFM3-Omni-2B-GGUF
artlite-cli -m $M/LFM3-Omni-2B-Q4_K_M.gguf --mmproj $M/mmproj-LFM3-Omni-2B-Q8_0.gguf --image photo.jpg --audio question.wav
artlite-cli -m $M/LFM3-Omni-2B-Q4_K_M.gguf --mmproj $M/mmproj-LFM3-Omni-2B-Q8_0.gguf --video clip.mp4 -p "What is said?"
artlite-cli -m $M/LFM3-Omni-2B-Q4_K_M.gguf --mmproj $M/mmproj-LFM3-Omni-2B-Q8_0.gguf --audio question.wav --speak reply.wav
artlite-server -m $M/LFM3-Omni-2B-Q4_K_M.gguf --mmproj $M/mmproj-LFM3-Omni-2B-Q8_0.gguf
```

```
$ARTLITE_MODELS_DIR/LiquidAI/LFM3-Omni-2B-GGUF/   LiquidAI/LFM3-Omni-2B-GGUF
    LFM3-Omni-2B-Q4_K_M.gguf                      (its language model: LFM3-1.7B's shapes, its own weights; Q4_0 too)
    mmproj-LFM3-Omni-2B-Q8_0.gguf                 (its vision encoder, speech encoder and speech generator; BF16, F16 too)
```

LiquidAI's LFM3-Omni-2B is LFM3's trunk with all three media networks in one mmproj: LFM3-VL's vision encoder (its own
weights), LFM3-Audio's speech encoder and speech generator. Everything the sections above describe runs on it:
pictures (64 to 2048 tokens: the mmproj's bounds), videos with their soundtrack ([Videos](#videos): at most 384
frames), audio clips, transcription (the offline ASR prompt; live transcription is LFM3-Audio's), text to speech,
spoken replies and realtime conversations ([artlite-server](../server.md#speech)), the template's tool calls. The three
networks share one vision overlay, so an image and a clip run back to back without switching hardware contexts (the
process holds three: the media's, the language model's decode overlay and its prefill library). Its mmproj must be
Q8_0, BF16 or F16: the vision overlay's GEMMs read bfp16, so a 4-bit mmproj would grow to twice its size at every
load (it is refused, naming Q8_0).

Accuracy, teacher-forced over llama.cpp's answers (KL of the next-token distributions against llama.cpp's language
model on exact (float64) media embeddings; llama.cpp's CPU encoders on the Q8_0 mmproj are themselves 3-4% from them):

| prompt | the NPU's language model on those embeddings | the NPU's encoders too | llama.cpp's own run |
|---|---|---|---|
| a picture | 0.0033 | 0.0051 | 0.0040 |
| a spoken clip (ASR) | 0.0002 | 0.0004 | 0.0002 |
| a picture with a spoken question | 0.0021 | 0.0035 | 0.0024 |
| a 14 s video with its soundtrack (7.3K tokens) | 0.0031 | 0.0068 | 0.0051 |

(The greedy answers the same; a video's error is its pairs': the tower's tokens 5.1% from float64 at the median,
llama.cpp's 3.2%.) Word error rates of the transcriptions (LibriSpeech, as [above](#lfm3-audio-2b-speech)): Artlite
2.27% / 4.87% (test-clean / test-other), llama.cpp on the CPU 2.22% / 4.93%. Its speech (200 LibriSpeech sentences spoken twice each, transcribed back by
Whisper base.en): 3.17% word error rate, llama.cpp's (CPU) 3.40% (paired: -0.23 points, 95% interval -0.64 .. +0.11). Questions
about videos (60 of Daily-Omni's multiple-choice questions, one per clip of 30 or 60 s, their sound included, greedy):
41 right, llama.cpp on the CPU 42, the same answer on 59 (the prompts' tokens the same).

The time to the first token of a media prompt through each runtime's server (Strix Halo, HTTP included; the media
fresh each request; energy: the SoC's power above an 18 W idle until the first token):

| prompt (prompt tokens) | Artlite (NPU), Q4_K_M / Q4_0 | llama-server, iGPU (Vulkan) | llama-server, CPU (16 threads) |
|---|---|---|---|
| a 512 x 512 picture with a 2 s spoken question (316) | 0.19 / 0.18 s, 1.7 J | 0.30 s, 9.0 J | 0.64 s, 34 J |
| a 10 s 640 x 360 video with its sound (2445) | 0.86 / 0.81 s, 16 J | 2.6 s, 101 J | 5.6 s, 293 J |
| a 30 s 720p video with its sound (10.7K) | 3.85 / 3.66 s, 134 J | 12.4 s, 470 J | 31 s, 1583 J |
| a 3-minute 720p video with its sound (14.5K) | 5.72 / 5.43 s, 236 J | 23.8 s, 915 J | 80 s, 3345 J |

(The 30 s and 3-minute videos: cuts of one cooking video copied from its stream, so FFmpeg converts all their frames
(llama.cpp's command); the same cuts re-encoded, as most files are, have only their sampled frames converted, but the
3-minute one's frames come out a little larger (pairs of 66 tokens: a run each): 4.25 and 6.54 s on Q4_K_M.) Most of a
long video's time is its pairs on the vision overlay and the prompt's prefill (14.5K tokens: 2.7 s); FFmpeg decodes
meanwhile.

Speech output, as LFM3-Audio's ([Speech output](#speech-output); `/tts`, streamed, warm):

| | 9 s of speech | 41 s of speech | first audio | energy per minute of speech |
|---|---|---|---|---|
| Artlite (NPU), Q4_K_M | 3.8 s | 16.5 s | 0.19 s | 150 J |
| Artlite (NPU), Q4_0 | 3.9 s | 12.0 s (30 s of speech) | 0.18 s | 150 J |
| llama.cpp, CPU (16 threads) | 2.1 s | 9.9 s | 0.2 s | 760 J |
| llama.cpp, iGPU (Vulkan) | 1.9 s | 8.2 s | 0.11 s | 420 J |

A realtime conversation's turn ([artlite-server](../server.md#speech): the audio runs on the model as it arrives)
answers in speech 0.36 s after the user's audio ends (`output_modalities: ["audio"]`; interleaved text and speech:
0.48 s, its first 6 text tokens before the first frames).

## d1-2B (decision model)

LiquidAI's d1-2B is a decision model on LFM3's trunk: typed answers to questions about a state (a choice, a score, a
yes / no probability) from one forward pass, no generated text, through
[artlite-server's `/v1/systemone`](../server.md#decisions), about text, pictures and videos (their frames; its mmproj:
LFM3-VL's encoder).

```
$ARTLITE_MODELS_DIR/LiquidAI/d1-2B-GGUF/         LiquidAI/d1-2B-GGUF
    d1-2B-Q4_0.gguf
    d1-2B-Q4_K_M.gguf
    d1-2B-Q8_0.gguf                              (the nearest to the BF16 model)
    mmproj-d1-2B-Q8_0.gguf                       (its vision encoder: pictures in requests; or its BF16, F16)
```

```bash
artlite-server -m LiquidAI/d1-2B-GGUF/d1-2B-Q4_0.gguf --mmproj LiquidAI/d1-2B-GGUF/mmproj-d1-2B-Q8_0.gguf
curl http://127.0.0.1:8080/v1/systemone -H "Content-Type: application/json" -d '{"state": "I was charged twice
  this month, please refund one of them.", "questions": {"refund": {"type": "noul", "instructions": "Is the customer
  asking for a refund?"}, "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria":
  {"billing": "Charges, refunds", "technical": "App faults"}}}}'
```

A request's pictures (`images`, or a chat state's `image_url` parts: [Decisions](../server.md#decisions)) head its
state, each its tokens between `<|vision_start|>` and `<|vision_end|>`, as d1's runner and llama-server place them.
A picture takes 64 to 1600 tokens of 32 x 32 pixels (d1's runner's budget, which it was trained at; llama-server reads
up to 4096: `--image-max-tokens 1600` there gives the same sizes); a state with its pictures fits one chunk of 4096
tokens. A video (a data URI, a chat state's `video_url` part) is its frames' pairs, as d1's runner reads them
([Videos](#videos)), without its soundtrack.

LiquidAI's d1-Omni-2B is d1 with LFM3-Omni's encoders (`LiquidAI/d1-Omni-2B-GGUF`: `d1-Omni-2B-Q8_0.gguf`, its mmproj
`mmproj-d1-Omni-2B-Q8_0.gguf`): a request's media can also be audio clips (data URIs, `input_audio` parts; a video
still its frames alone, as d1's runner reads it), and a question can be spoken (its `instructions` an `input_audio`
part: heard where its text would be), as llama-server reads them. Against llama-server on the CPU, each question
alone: the same answer on 316 of the 320 text questions above (the other four near ties, within 0.02), the largest
difference in a probability 0.033 (median 0.004); about media 8 of 9 (the other a near tie), within 0.042. Through the
servers (fresh media each request; energy above an 18 W idle), a request about a 14 s clip (4 questions), a picture
with a spoken question (2) or a 4 s video's frames (3): 157 / 183 / 203 ms, 0.4 / 1.1 / 0.8 J a question (Artlite:
its questions in one sequence); llama-server on the iGPU 362 / 317 / 701 ms, 3.3 / 4.9 / 6.7 J; on the CPU 595 /
846 / 1287 ms, 7.8 / 21 / 21 J.

How it runs: its full-attention layers are non-causal, so a request is read as d1's own runner reads it, the state
once and up to 16 questions after it in one sequence (a request's `set_size`: [Decisions](../server.md#decisions)),
in one prefill chunk from position 0 (at most 4096 tokens). The chunk's full layers use the prefill attention's
non-causal design (`prefill/attn.py`): every pass streams the sequence's cache groups, the state's tokens see every
token, a question's tokens the state and their own question (per row its own keys, in an object per pass; the chunks
no row of a core sees skipped). The sliding-window and conv layers stay causal, the conv from zero histories. The
answers come from each question's last row and its options' label tokens' rows of the LM head, on the host. A
request's pictures run through the vision encoder first (LFM3-VL-2B's: [above](#lfm3-vl-2b-images)), once each; their
tokens are the state's.

Speed (Strix Halo, through the servers, HTTP included, warm; energy per question above a ~20 W idle):

| request (Q4_0) | NPU | iGPU (llama-server Vulkan) | CPU (llama-server) |
|---|---|---|---|
| a question of 64 / 256 / 1024 prompt tokens | 33 / 49 / 149 ms | 22 / 69 / 298 ms | 37 / 81 / 293 ms |
| energy | 0.36 / 0.86 / 3.3 J | 0.73 / 2.4 / 10 J | 1.5 / 3.0 / 12 J |
| 8 questions about one state (~150 words) | 84 ms (one sequence of 482 tokens; each alone: 370 ms) | 490 ms (each alone: 1833 tokens) | 650 ms |
| energy a question | 0.23 J | 1.8 J | 3.4 J |

Accuracy against llama-server on the CPU, on the same 320 questions (50 each of SST-2, AG News, BoolQ, MNLI, and 120
about essays), each question alone as llama-server reads them:

| | Q4_0 | Q8_0 |
|---|---|---|
| the same answer (a noul's side) | 318 / 320 | 320 / 320 |
| largest difference in a probability (median) | 0.047 (0.003) | 0.030 (0.003) |
| accuracy SST-2 / AG News / BoolQ / MNLI | 98 / 86 / 88 / 96% (llama-server 98 / 86 / 90 / 94%) | 98 / 84 / 90 / 92% (the same) |

Questions in one sequence answer as d1's runner does (its Hugging Face model in fp32: the fp32 evaluation of the Q8_0
file is within 0.012 of it on 27 requests of 4-8 questions, with the same prompts token for token), which differ from
each question alone by up to 0.27 in a probability (median 0.09 a request).

With pictures (57 questions on 16 requests: photos of 128 to 1600 tokens, drawn shapes, an invoice; one or two
pictures with a state, pictures alone, a chat state), each question alone against llama-server on the CPU (its
mmproj on the CPU too): Q4_0 57 / 57 the same answer, the largest difference in a probability 0.021 (median 0.003);
Q8_0 56 / 57 (a score's near tie), 0.029 (0.003); the prompts' tokens the same as llama-server's. Speed, one question
about a picture and 8 about one (energy a question above a 16.5 W idle, the three in one session):

| request (Q4_0 + mmproj Q8_0) | NPU | iGPU (llama-server Vulkan) | CPU (llama-server) |
|---|---|---|---|
| a question about a 512 x 512 / 1024 x 768 / 1280 x 1280 picture (292 / 802 / 1633 tokens) | 86 / 274 / 591 ms | 118 / 385 / 964 ms | 488 / 1747 / 3215 ms |
| energy | 1.5 / 3.9 / 8.0 J | 4.2 / 12 / 26 J | 16 / 60 / 115 J |
| 8 questions about a 512 x 512 picture | 126 ms (one sequence of 516 tokens; each alone: 646 ms) | 1184 ms | 3667 ms |
| energy a question | 0.29 J | 3.8 J | 16 J |

Design: the module docstrings of `decode/design.py`, `decode/attn.py` and `lfm/`.
