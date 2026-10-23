# Benchmark methodology

The README's charts compare Artlite with FastFlowLM and OpenFlowLM (FastFlowLM's fork on open NPU kernels) on one
machine, with the same prompts, wherever they run the same model with comparable 4-bit weights, and an image
prompt's time to first token also with llama.cpp on the iGPU:

| chart | what |
|---|---|
| decode | tokens/s generated after a 1k- and a 32k-token prompt; Gemma 4 and Qwen3.6 35B-A3B also with their drafters (MTP) |
| prefill | tokens/s of a 1k- and a 32k-token prompt |
| TTFT | time to first token against prompt length, 1k to 128k tokens, log-log |
| image TTFT | MiniCPM-V 4.7 1B's time to first token with an image, four images, its 16x and 4x vision encoders |

Each Artlite bar is one GGUF file ([below](#measured-files)). FastFlowLM's and OpenFlowLM's bars stand under the
Artlite file with comparable weights: their q4nx files hold a Q4_1 or Q4_K GGUF's numerics in their own tiles.

## Test system

| | |
|---|---|
| machine | ASUS ROG Flow Z13 (2025, GZ302EA), BIOS 314 |
| processor | AMD Ryzen AI Max+ 395: 16 Zen 5 cores, Radeon 8060S iGPU, XDNA 2 NPU |
| memory | 128 GB LPDDR5X-8000, 256-bit (256 GB/s) |
| OS | Gentoo Linux, kernel 7.2.9 |
| NPU driver and firmware | `amdxdna` from AMD's xdna-driver (2.21.70-790-g49fe4e13), NPU firmware 1.1.2.65 (xdna-driver's build, `npu.dev.sbin`; linux-firmware's `npu_7.sbin` 1.1.2.65 measures the same) |
| power | ACPI platform profile `performance` in both of its handlers (amd-pmf and asus-wmi: `/sys/class/platform-profile/*/profile`; `/sys/firmware/acpi/platform_profile` reads `custom` when they differ); NPU power mode: the driver's default for Artlite (it sets none), `turbo` for FastFlowLM and OpenFlowLM |

One NPU job at a time, with no other heavy CPU load.

## Artlite

| number | how |
|---|---|
| TTFT | `artlite-bench -p 1024,8192,32768,65536 -n 0 -r 3` (MiniCPM-V 4.7 1B also 131072): `avg_ns` of each `pp<n>` (the prompt's chunks and the last token's logits) |
| prefill | the same run: `avg_ts` of `pp1024`, `pp32768` |
| decode | `artlite-cli -n 256 --greedy` on a prompt of 1023 / 32767 tokens: its generation speed |
| decode with MTP | the same at the best tokens per step for each length (1 to 4) |
| image TTFT | `artlite-server --mmproj FILE`, streaming `/v1/chat/completions` requests of an image and "Describe this image in detail.", thinking off, greedy: the client's time to the first token, median of 3 after a warmup |

The prompts are Paul Graham's essays (`tests/datasets/paul_graham_essays` in file-name order, fetched by
`tests/datasets/fetch_datasets.py`) followed by a request to summarize them, cut to the length with the model's chat
template. Decode is greedy, so the text, and with it the drafts' acceptance, is the same on every run; it does not come
from `artlite-bench`'s `tg` tests, whose random tokens a drafter never guesses. The model file is read into the page
cache first, and the designs are compiled before the timed runs.

The image prompts' images: `docs/images/sd15-astronaut.jpg` (512 x 512), OpenBMB/MiniCPM-V's `assets/airplane.jpeg`
(1920 x 1440) and `assets/hk_OCR.jpg` (1300 x 858), and llama.cpp's `tools/mtmd/test-1.jpeg` (640 x 488). Each is cut
as the official processor cuts it, into an overview and up to 9 slices, in every runtime. No request is served from a
cache: each one sends the image's pixels with a random 4 x 4 block in its corner (re-encoded as JPEG quality 95) and a
random word before the question. The same client measures every runtime's server.

The step-by-step procedure, with its prompt and draft-count scripts, is the
[artlite-bench skill](../.claude/skills/artlite-bench/SKILL.md).

### Measured files

| model | quant | file | sha256 | recorded |
|---|---|---|---|---|
| Llama 3.2 1B Instruct | Q4_0 | [unsloth/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-Q4_0.gguf) `Llama-3.2-1B-Instruct-Q4_0.gguf` | `66bfbb2d48bd` | 2026-10-15 |
| | Q4_1 | [unsloth/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-Q4_1.gguf) `Llama-3.2-1B-Instruct-Q4_1.gguf` | `a8bfc798aceb` | 2026-10-15 |
| | Q4_K_M | [unsloth/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf) `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | `3f5a22426976` | 2026-10-15 |
| | Q3_K_L | [lmstudio-community/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/lmstudio-community/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-Q3_K_L.gguf) `Llama-3.2-1B-Instruct-Q3_K_L.gguf` | `b5019ccdf126` | 2026-10-15 |
| | BF16 | [unsloth/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-BF16.gguf) `Llama-3.2-1B-Instruct-BF16.gguf` | `f5d05e90d179` | 2026-10-15 |
| | Q6_K | [bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-Q6_K.gguf) `Llama-3.2-1B-Instruct-Q6_K.gguf` | `0f4c510daf16` | 2026-10-15 |
| | Q5_K_M | [MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct.Q5_K_M.gguf) `Llama-3.2-1B-Instruct.Q5_K_M.gguf` | `a8c8692a3f2d` | 2026-10-15 |
| Llama 3.2 3B Instruct | Q4_K_M | [unsloth/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf) `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | `6c99cc00ae91` | 2026-10-21 |
| | BF16 | [unsloth/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-BF16.gguf) `Llama-3.2-3B-Instruct-BF16.gguf` | `9b8dce13b6cb` | 2026-10-15 |
| | Q4_0 | [bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q4_0.gguf) `Llama-3.2-3B-Instruct-Q4_0.gguf` | `4d491003bf6b` | 2026-10-15 |
| | Q5_K_M | [bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q5_K_M.gguf) `Llama-3.2-3B-Instruct-Q5_K_M.gguf` | `0b94ccd04d90` | 2026-10-15 |
| | Q2_K | [MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct.Q2_K.gguf) `Llama-3.2-3B-Instruct.Q2_K.gguf` | `436c90a401ec` | 2026-10-15 |
| | Q3_K_M | [MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct.Q3_K_M.gguf) `Llama-3.2-3B-Instruct.Q3_K_M.gguf` | `1399c90c682c` | 2026-10-15 |
| Llama 3.1 8B Instruct | Q4_K_M | [lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf) `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | `f2be3e1a239c` | 2026-10-15 |
| | Q4_K_S | [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf) `Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf` | `7b04439ae150` | 2026-10-15 |
| Gemma 4 E4B it | Q4_0 | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q4_0.gguf) `gemma-4-E4B-it-Q4_0.gguf`, its drafter `mtp-gemma-4-E4B-it-Q4_0.gguf` | `a555b900214b`, `d19a11e4fd50` | 2026-10-15 |
| Qwen3.8 27B | Q4_K_M | [lmstudio-community/Qwen3.8-27B-GGUF](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q4_K_M.gguf) `Qwen3.8-27B-Q4_K_M.gguf` | `e00082f779fa` | 2026-10-21 |
| Qwen3.6 35B-A3B | Q4_K_M | [lmstudio-community/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/lmstudio-community/Qwen3.6-35B-A3B-GGUF/blob/main/Qwen3.6-35B-A3B-Q4_K_M.gguf) `Qwen3.6-35B-A3B-Q4_K_M.gguf`, its drafter [ggml-org/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/blob/main/mtp-Qwen3.6-35B-A3B-Q4_0.gguf) `mtp-Qwen3.6-35B-A3B-Q4_0.gguf` | `4ac6a06bce55`, `606fca331adc` | 2026-10-24 |
| MiniCPM-V 4.7 1B | Q4_K_M | [openbmb/MiniCPM-V-4.7-1B-gguf](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-gguf/blob/main/MiniCPM-V-4_7-1B-Q4_K_M.gguf) `MiniCPM-V-4_7-1B-Q4_K_M.gguf` (the language model), its vision encoders [`mmproj-model-f16.gguf`](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-gguf/blob/main/mmproj-model-f16.gguf) (16x) and [`mmproj-model-f16-4x.gguf`](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-gguf/blob/main/mmproj-model-f16-4x.gguf) (4x) | `588c6c94c82f`, `fb9cc047a3cd`, `1e0ab439877e` | 2026-10-21 |

sha256: the first 12 hex digits of the file's, the same as Hugging Face's LFS object id. Recorded: the date of the
runs, with the toolchain pinned then:

| recorded | mlir-aie | llvm-aie (Peano) |
|---|---|---|
| 2026-10-15 | 1.4.5.dev30+g766aca4 | 22.0.0.2026101201+83ba9ea9 |
| 2026-10-21 | 1.4.5.dev56+g1a647f1 | 22.0.0.2026102001+4f5df13d |
| 2026-10-22 | 1.4.5.dev56+g1a647f1 | 22.0.0.2026102001+4f5df13d |

## FastFlowLM and OpenFlowLM

FastFlowLM 1.0.7 (v1.0.7-56-g6c9ed874; 1.0.9, v1.0.9-41-gd8767c0a, for Qwen3.8 27B and MiniCPM-V 4.7 1B) and OpenFlowLM 0.1.0-99fe7e4
(Atomic-Germ/OpenFlowLM-Next at 99fe7e4, every model on its open kernels), each as `serve` with its default prefill
chunks of 4096 tokens and each model at its native context:

- TTFT: a streaming `/v1/chat/completions` request with the same essay prompt of n tokens, greedy, 128 tokens: the
  client's time to the first token, median of up to 3;
- prefill: n / TTFT;
- decode: (tokens - 1) / the time from the first to the last streamed token;
- image TTFT (FastFlowLM 1.0.9): Artlite's image prompts and client, 16 tokens. Its MiniCPM-V has the 16x vision
  encoder only (bf16).

| runtime | model | its model | weights | under Artlite's |
|---|---|---|---|---|
| FastFlowLM | Llama 3.2 1B Instruct | `llama3.2:1b` | Unsloth's imatrix Q4_1 GGUF, the Q4_1 file measured above | Q4_1 |
| | Llama 3.2 3B Instruct | `llama3.2:3b` | imatrix Q4_1 GGUF (Unsloth style) | Q4_K_M |
| | DeepSeek R1 Distill Llama 8B (Llama 3.1 8B's shapes) | `deepseek-r1:8b` | imatrix Q4_1 GGUF, untied | Llama 3.1 8B Q4_K_M |
| | Gemma 4 E4B it | `gemma4-it:e4b` | imatrix Q4_1 GGUF; int8 embeddings, bf16 per-layer projections | Q4_0 |
| | Qwen3.8 27B | `qwen3.8-mtp:27b` | Q4_K numerics re-fit to FastFlowLM's Q4_K (4.625 bpw); lm_head and embedding Q8_0; its MTP layer (7 drafts per step); at most 8192 positions | Q4_K_M |
| | Qwen3.6 35B-A3B | `qwen3.6-moe:35b-a3b` | Q4_K numerics re-fit to FastFlowLM's Q4_K (4.625 bpw) | Q4_K_M |
| | MiniCPM-V 4.7 1B | `minicpm-v-4.7:1b` | FastFlowLM's own Q4_K quantization (560 MB per token, the lm_head Q8_0), not openbmb's GGUF files; text prompts | Q4_K_M |
| OpenFlowLM | Llama 3.2 1B Instruct | `llama3.2:1b` | FastFlowLM's container of the same Q4_1 | Q4_1 |
| | Llama 3.2 3B Instruct | `llama3.2:3b` | imatrix Q4_1 GGUF (FastFlowLM's container) | Q4_K_M |
| | Llama 3.1 8B Instruct | `llama3.1:8b` | imatrix Q4_1 GGUF (FastFlowLM's container) | Q4_K_M |
| | Qwen3.8 27B | `qwen3.8:27b` | Q4_1 linears, Q8_0 lm_head and ssm_out, converted from a Q8_0 GGUF; no MTP head; 131072 positions | Q4_K_M |

OpenFlowLM runs that take over 5 minutes were not measured; the charts mark the missing values. The data files also
hold models with no Artlite row yet (FastFlowLM's Qwen3 0.6B, Gemma 4 E2B; OpenFlowLM's Qwen3 0.6B, Qwen3.5 9B),
which the charts leave out.

## llama.cpp

llama.cpp b11699 (7182ef71e, built with Vulkan and `GGML_NATIVE`) on the Radeon 8060S iGPU, Mesa 26.2.4 (RADV), for
the image TTFT chart: `llama-server -m MiniCPM-V-4_7-1B-Q4_K_M.gguf --mmproj FILE -c 32768 -np 1 -ngl 99`, the model
and its vision encoder on the iGPU (flash attention on), the same GGUF files, image prompts and client as Artlite's,
`cache_prompt` false. The iGPU's sessions (a server start and its requests) differ by up to 8%: the chart has the best
of 3.

## Data and charts

| file | |
|---|---|
| `data/artlite.json` | Artlite's results: per model and quant, `ttft_s`, `prefill_tps`, `decode_tps` (keys: prompt lengths in tokens), `mtp`, `image_ttft_s` (keys: the vision encoder, then the image), `notes` (footnotes); `image_prompts`: the question, the encoders and the images (size, slice grid) |
| `data/fastflowlm.json`, `data/openflowlm.json`, `data/llama.cpp.json` | the other runtimes' results in the same schema; `comparable_quants` pairs a row with Artlite's files, `same_shapes_as` charts it under a model of the same architecture and sizes |
| `charts.py` | draws `images/{decode,prefill,ttft,image-ttft}.svg` and their `-dark` variants from the data |

```bash
python docs/charts.py
```

Each model page has more detailed measurements: [Models](models/README.md).
