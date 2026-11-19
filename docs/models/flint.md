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
$ARTLITE_MODELS_DIR/incoai/Flint-DFlash2-GGUF/         incoai/Flint-DFlash2-GGUF
    Flint-DFlash2-Q4_K_M.gguf                          (its DFlash2 drafter, optional, the default where it is here)
```

Other Flint files (unsloth's UD-Q4_K_XL, lmstudio-community's Q4_K_M: other tile formats) are refused for now. The
models directory: [Installation](../install.md#models-directory).

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
each MoE layer, 1.46 GB in all. The KV caches take 84.7 KB per position (K as hi + lo blocks); past 16384 positions of
capacity the sliding-window layers' caches are rings of 16384 positions (1.0 GB) and the full layers' take 24.2 KB per
position: 1.8 GB at 32K, 7.4 GB at 256K (22.2 GB as linear caches); 24.3 GB of RAM at a 4K context. With the rings a
conversation goes back only to a checkpoint (artlite-server takes them: the rings' windows, ~35 MB each).

## Drafting

Generation drafts with incoai's DFlash2 block drafter where its file is in the models directory (`--spec-type none`:
without it; `artlite/decode/dflash/`). One dispatch of the drafter's own design, in a hardware context of its own,
guesses a block of tokens after the next one from the model's rows at five layers of its last step (the drafter's
taps); the model's next step checks the first 3 of them with the token (steps of up to 4 slots) and keeps the ones it
agrees with, so the text is the one the model writes alone. The drafter's head is the first 33280 rows of the model's
LM head, laid out for it at load (the head's rows reordered, no weight copied). Each slot of a step routes its own
experts (~+33% of a one-token step), so a round's step takes only the drafts whose probability (the drafter's, as the
rounds calibrate it) pays for their slots at the measured step times, and the generation runs the drafter's rounds
while they beat steps of one token. Short chat prompts, 400 tokens greedy, against ~32.4 tokens/s without it: math
1.31x, code 1.24x, prose 1.13x, JSON 1.07x; the drafter takes 0.19 GB.

Several requests at a time (`artlite-server -np N`): the NPU holds N sequences' KV caches, and a decode step's 4 slots
are tokens of the requests that generate, each in its own cache (`models/lanes.py`). The dense weights stream once for
all of them, and an MoE layer runs the union of their experts: a step of 2 / 3 / 4 requests takes 1.37x / 1.84x / 2.24x
a step of one, so 4 requests get 1.4x one request's tokens per second in all. The DFlash2 drafter drafts for a
request that generates alone (37.4 tokens per second on a server with `-np 4`); its records are one sequence's at a
time, saved and brought back as the requests change (11 MB at 1K positions), so where requests share the steps it
stands back and their rows wait for it: [artlite-server](../server.md#several-requests-at-a-time).

## Speed and accuracy (Strix Halo)

| | 1K | 8K | 32K | 64K |
|---|---|---|---|---|
| time to the first token (`artlite-bench`, random tokens) | 0.81 s | 3.76 s | 21.5 s | 61.4 s |
| generation (greedy, 256 tokens, essay summary) | 31.4 tokens/s | | 19.3 tokens/s | |
| the same with the DFlash2 drafter (steps of up to 4 tokens) | 36.5 tokens/s | | 21.6 tokens/s | |

A 128K prompt: 197 s to the first token (the full layers' attention is most of it past 64K).

Accuracy against llama.cpp's CPU runs of the same file:

| | |
|---|---|
| teacher-forced decode, an essay's first 128 tokens | KL 0.019 (llama.cpp's own prompt graph against its decode: 0.014), argmax the same on every clear token |
| a prompt through the prefill, then 24 tokens | 2100-token prompt: KL 0.0135 (llama.cpp's decode-only run: 0.0128) |

An MoE model's routing turns tiny rounding differences into another expert set for some tokens (llama.cpp's own prompt
and decode graphs choose another top 6 in 21-24% of the tokens' layers), so its KL against any other implementation stays
near llama.cpp's own.

Design: the module docstrings of `src/models/flint/`, `src/artlite/decode/attn.py` and `src/artlite/moe/`.
