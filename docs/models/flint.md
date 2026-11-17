# Flint on the NPU

Recursive's Flint (GGUF `flint`): 40B parameters in 49 layers, a mixture of 256 experts of which 6 run per token (~2.5B
active) beside a shared expert, sliding-window attention layers of 513 keys and full ones without positional encoding,
learned attention sinks. The architecture, the tokenizer and the chat template come from the file's metadata. Running
it: [artlite-cli](../cli.md), [artlite-server](../server.md); default sampling: the file's (temperature 1, top-p 0.95,
top-k 20).

## Model files

```
$ARTLITE_MODELS_DIR/RecursiveSI/flint-int4-gguf/       RecursiveSI/flint-int4-gguf
    flint-int4-q4_1.gguf                               (the official file: the model's own INT4 codes in groups of 64,
                                                        stored as pairs of Q4_1 blocks; F32 router, norms, sinks)
```

The models directory: [Installation](../install.md#models-directory).

## Thinking and tools

The model opens its thought (`<think>`) by itself; the chat template has no switch. The thought comes back as
`reasoning_content` (artlite-server), and earlier turns' thoughts stay out of the prompt unless a request sets the
template variable `preserve_reasoning`. Tool calls use GLM-4.5's syntax (`<tool_call>name<arg_key>..</arg_key>
<arg_value>..</arg_value></tool_call>`), parsed into OpenAI's `tool_calls`; tool results go back as `role: tool`
messages (the template's `<|observation|>` turns). A generation ends at `<|end_of_text|>`, `<|user|>` or
`<|observation|>`.

## How the NPU path works

Paths below are under `src/models/flint/` and `src/artlite/` (what the models share). One design, compiled once: the
decode overlay of 4 token slots (`artlite.decode`) with the attention cores of Gemma, Llama, Qwen3 and LFM3
(`decode/attn.py`) in Flint's configuration and the MoE layers of Qwen3.6 35B-A3B (`artlite.moe`).

- A dispatch runs up to 4 token slots through every layer and the LM head, streaming the weights once for all of them
  (`overlay.py`, `FlintOverlay`): the projections on each column's main and helper cores; the attention on its own
  cores, two columns per KV head with its 8 query heads, the sinks added to the softmax's sum; each MoE layer's router
  on the helper cores, which push their columns' experts' weights themselves, so the host never sees the routing.
- The prompt runs on a prefill (`prefill.py`, `FlintPrefill`): chunks of up to 8192 tokens as matrix products on the same
  weights, the attention of each layer kind, and per MoE layer the router, the host's routing of the chunk's tokens and
  the experts' work lists, writing the caches the decode continues from.

Memory: the weights the NPU streams are re-laid-out into device buffers at load, in memory, at their quantization's
size: 22.26 GB (each pair of Q4_1 blocks shares one scale and offset: 144 bytes per row and 256 columns instead of 160,
no weight changed); nothing is written to disk. A decode step streams 0.97 GB of dense weights and the 6 experts of
each MoE layer, 1.46 GB in all. The KV caches take 84.7 KB per position (K as hi + lo blocks; the sliding-window layers'
caches grow with the positions): 2.8 GB at 32K; 24.3 GB of RAM at a 4K context.

## Speed and accuracy (Strix Halo)

| | 1K | 8K | 32K | 64K |
|---|---|---|---|---|
| time to the first token (`artlite-bench`, essays) | 0.81 s | 3.87 s | 23.7 s | 70.1 s |
| generation (greedy, 256 tokens, essay summary) | 31.4 tokens/s | | 19.3 tokens/s | |

Accuracy against llama.cpp's CPU runs of the same file:

| | |
|---|---|
| teacher-forced decode, an essay's first 128 tokens | KL 0.019 (llama.cpp's own prompt graph against its decode: 0.014), argmax the same on every clear token |
| a prompt through the prefill, then 24 tokens | 2100-token prompt: KL 0.0135 (llama.cpp's decode-only run: 0.0128) |

An MoE model's routing turns tiny rounding differences into another expert set for some tokens (llama.cpp's own prompt
and decode graphs choose another top 6 in 21-24% of the tokens' layers), so its KL against any other implementation stays
near llama.cpp's own.

Design: the module docstrings of `src/models/flint/`, `src/artlite/decode/attn.py` and `src/artlite/moe/`.
