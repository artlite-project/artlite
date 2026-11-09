# LFM3 on the NPU

LiquidAI's LFM3-1.7B from a GGUF file (architecture `lfm3`): 24 layers of three kinds, 13 short-convolution layers, 6
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
| time to first token, 1K / 8K / 32K / 64K tokens | 0.153 / 1.21 / 6.26 / 16.3 s | 0.174 / 1.37 / 6.89 / 17.5 s |
| prefill, 1K / 32K tokens | 6698 / 5236 tokens/s | 5869 / 4754 tokens/s |
| generation after 1K / 32K tokens | 50.0 / 41.5 tokens/s | 47.6 / 39.8 tokens/s |
| a step of 2 / 3 / 4 tokens against one | +2% / +5% / +6% (20.2 ms for one at 1K) | |
| teacher-forced KL against the fp32 evaluation (a chat / an essay; llama.cpp's own on the same file) | 5.7e-4 / 5.5e-4 (7.0e-4 / 6.3e-4) | 1.1e-3 / 8.7e-4 (5.1e-3 / 2.5e-3) |
| the same after a prompt through the prefill | 1.1e-3 / 7.3e-4 (the prefill's 8-bit GEMM operands) | 1.3e-3 / 1.0e-3 |
| tool calls: BFCL v3 simple / multiple / parallel, 50 each, thinking off | 140 / 150 (llama-server on the same file: 140) | |
| GSM8K, the first 150 test questions, greedy, thinking off | 78.0% (llama.cpp on the CPU, the same requests: 80.7%; 7 questions right on the NPU alone, 11 on the CPU alone) | |

Q8_0: KL 6.2e-4 / 5.7e-4 (llama.cpp 1.0e-3 / 5.2e-4). Steps of 1-4 tokens give the same logits bit for bit, and a greedy
text with the context's drafts is the text without them.

Against FastFlowLM 1.0.12 on the same NPU (its file of the same QAD Q4_0 weights, at most 32K positions): the first token
8.4x sooner at 1K tokens (1.28 s there), 4.5x at 32K (28.2 s); generation 1.2x / 1.8x as fast after 1K / 32K (41.1 / 23.6
tokens/s). Its documentation reports 81.3% on the first 150 GSM8K questions (its own requests) against its CPU engine's
87.3%, the difference its NPU attention's.

Against llama.cpp's Vulkan backend on the iGPU (the same QAD-Q4_0 file): a prompt runs 1.16x as fast at 1K tokens and
1.9x at 32K, with ~3.3x less energy; generation runs at ~30% of its speed (168.8 / 122.5 tokens/s after 1K / 32K):
[Where the NPU fits](../npu-vs-igpu.md#a-small-hybrid-models-prompts-lfm3-17b).

Design: the module docstrings of `decode/design.py`, `decode/attn.py` and `lfm/`.
