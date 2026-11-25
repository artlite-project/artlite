# Inkling 2 on the NPU

Thinking Machines' Inkling 2 Mini (GGUF `inkling2_loop`, or unsloth's looped `inkling2`): 26B parameters in 24 layers,
a mixture of 128 experts of which 8 run per token, the middle 12 layers run 3 times (48 layer visits a token), each layer
mixing the earlier blocks' states (depth mixes) and, past the loop's first pass, its passes' outputs through learned
controllers. Attention: 40 of the 48 visits a sliding window of 1024 keys, the others full, K and V through short
convolutions, a relative-position bias, learned sinks. The architecture, the tokenizer and the chat template come from
the file's metadata. Running it: [artlite-cli](../cli.md), [artlite-server](../server.md); default sampling: the
file's (temperature 1, all tokens).

## Model files

```
$ARTLITE_MODELS_DIR/unsloth/Inkling-2-Mini-GGUF/        unsloth/Inkling-2-Mini-GGUF
    Inkling-2-Mini-UD-Q4_K_XL.gguf                      (NVFP4 and Q8_0 experts, Q8_0 attention, BF16 router, controllers)
```

The models directory: [Installation](../install.md#models-directory).

## Thinking and tools

The template puts a system block "Thinking effort level: x" before the first message (`reasoning_effort`: `none`
.. `max`, or a number): with [artlite-cli](../cli.md)'s `--think` the template's default (0.9) or `--reasoning-effort`'s
level, without it `none`. The model opens its own blocks (`<|message_model|><|content_thinking|>`, then
`<|content_text|>`); the thought comes back as `reasoning_content` (artlite-server). Tools are declared to the model as
the template writes them (one JSON line in a `tool_declare` block); artlite-server does not yet parse the model's calls
(`<|content_invoke_tool_json|>`) into `tool_calls`. A generation ends at `<|content_model_end_sampling|>`.

## How the NPU path works

Paths below are under `src/models/inkling2/` and `src/artlite/` (what the models share). One design, compiled once:
the decode overlay of 4 token slots (`artlite.decode`) with the attention cores in Inkling's mode (`decode/attn.py`)
and the MoE layers of `artlite.moe` with an expert's units of 512 and 128 channels.

- A dispatch runs up to 4 token slots through the 48 visits and the LM head, streaming the weights once for all of them
  (`overlay.py`, `InklingOverlay`): the projections on each column's main and helper cores, the residual stream's
  states (the blocks' outputs, the loop's banks) in the helpers' memory, the depth mixes and the controllers' blends
  there; the attention, its caches and the K / V convolutions on the attention cores; the experts' activation (situ)
  on the helpers; the LM head screened (the host computes its largest logits exactly).
- Prompts run in prefill chunks of 64 .. 8192 tokens (`prefill.py`, `InklingPrefill`, on the same weight image and
  caches): per visit the depth mixes, the loop's controllers, the attention (its prep, the relative bias and sinks on the
  attention cores; past 2048 tokens in items of 2048), the post convs, and the MoE (`moe.py`: the router on the NPU on x
  hi + lo, the routing on the host, the experts' work lists on `artlite.moe`); the prompt's last token runs as a decode
  step (the LM head's). The experts stream their weights once per chunk: long prompts run in chunks of 8192.

Memory: the weights the NPU streams are re-laid-out into device buffers at load, in memory, at their quantization's
size: 17.15 GB (a K of 2304 as two chunks of 1024 and a quarter of one, nothing padded); nothing is written to disk. A
decode step streams 1.09 GB of dense weights and the 8 experts of each visit (1.86 GB of expert tiles a token). The
sliding-window visits' caches are rings past 16K positions of capacity (16384 positions, two of the prefill's largest
chunks): they, the full visits' caches and the convs' state take 3.4 GB at 33K positions (each sequence of `-np` its
own), 6.2 GB at 132K. Without the prefill the rings hold 2048 positions (1.38 GB at 33K). The convs' state holds the last
positions only, so a conversation goes back only to a checkpoint (artlite-server takes them).

## Speed and accuracy (Strix Halo)

| | 1K | 32K |
|---|---|---|
| a prompt (the prefill, artlite-bench) | 472 tokens/s (2.17 s for 1024 tokens) | 578 tokens/s (56.7 s; 8K 12.8 s, 64K 132 s) |
| generation (greedy, 256 tokens, essay summary) | 10.7 tokens/s | 9.1 tokens/s |

A step of a token streams 1.09 GB of dense weights and its 8 experts per visit (1.86 GB of tiles); each further token of
a step (drafts, or other requests with `-np`) adds its own experts. Context drafts (artlite-cli's lookup) keep the text
the model writes alone.

Accuracy against llama.cpp's CPU runs of the same file:

| | |
|---|---|
| teacher-forced decode, an essay's first 128 tokens | KL 0.0016 (llama.cpp's own prompt graph against its decode: 0.0022), argmax the same on every clear token |
| a prompt through the prefill (1024, 2560 essay tokens), then 64 teacher-forced steps | KL 0.0020, 0.0023 (llama.cpp's own prompt graph against its decode: 0.0015), argmax the same on every clear token |
| greedy text, 128 tokens after a chat, code, math, a long instruction, an essay (prompts through the prefill) | 632 of 640 tokens llama.cpp's top one, the 8 others within 1 logit of it |

Design: the module docstrings of `src/models/inkling2/`, `src/artlite/decode/attn.py` and `src/artlite/moe/`.
