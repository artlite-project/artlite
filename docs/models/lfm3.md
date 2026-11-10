# LFM3 on the NPU

LiquidAI's LFM3-1.7B (and LFM3-VL-2B with images: [below](#lfm3-vl-2b-images)) from a GGUF file (architecture `lfm3`):
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

## d1-2B (decision model)

LiquidAI's d1-2B is a decision model on LFM3's trunk: typed answers to questions about a state (a choice, a score, a
yes / no probability) from one forward pass, no generated text, through
[artlite-server's `/v1/systemone`](../server.md#decisions), about text and pictures (its mmproj: LFM3-VL's encoder).
Video does not run yet.

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
tokens.

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
