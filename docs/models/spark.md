# Spark-X on the NPU

XHToken's Spark-X2.6 4B and 1.7B (and Spark-X2.5, the same architecture: GGUF `spark2_5`) from a GGUF file, with their
DSpark drafters. The architecture, the tokenizer and the chat template come from the file's metadata. Running it:
[artlite-cli](../cli.md), [artlite-server](../server.md); default sampling: the file's (temperature 1, top-p 0.95).

## Model files

```
$ARTLITE_MODELS_DIR/XHToken/Spark-X2.6-4B-GGUF/            XHToken/Spark-X2.6-4B-GGUF
    Spark-X2.6-4B-Q4_K_M.gguf                              (Q4_K, Q6_K and Q8_0 tensors, BF16 gates, a Q8_0 head)
    Spark-X2.6-4B-Q8_0.gguf
$ARTLITE_MODELS_DIR/XHToken/Spark-X2.6-4B-DSpark-GGUF/     XHToken/Spark-X2.6-4B-DSpark-GGUF
    Spark-X2.6-4B-DSpark.gguf                              (the 4B's drafter: found by the model's repository name)
$ARTLITE_MODELS_DIR/XHToken/Spark-X2.6-1.7B-GGUF/          (and -DSpark-GGUF: the same for the 1.7B)
$ARTLITE_MODELS_DIR/bartowski/Spark-X2.6-4B-GGUF/          bartowski/Spark-X2.6-4B-GGUF
    Spark-X2.6-4B-Q4_K_M.gguf, Spark-X2.6-4B-Q4_0.gguf     (a tied Q6_K head; Q4_K_M's gate / up in two formats)
```

The drafter of a model file is the first file of a repository `<the model's repository>-DSpark-GGUF` under any account
whose metadata fits the model (`--spec-type none`: none; `-md FILE`: another). The models directory:
[Installation](../install.md#models-directory).

## Thinking

The chat template thinks by default; `--think` / `--reasoning-effort high|low` in artlite-cli, a request's
`reasoning_effort` in artlite-server (`none` turns it off; clients' other levels map to the two: minimal and low to
low, medium, high and xhigh to high). Tool calls use GLM-4.5's syntax (`<tool_call>name<arg_key>..</arg_key>
<arg_value>..</arg_value></tool_call>`), parsed into OpenAI's `tool_calls`.

## How the NPU path works

Paths below are under `src/models/spark/` and `src/artlite/` (what the models share). One design, compiled once: the
decode overlay of 8 token slots (`artlite.decode`) with the head_dim-256 attention core shared with Qwen3.5
(`decode/attn256.py`) in Spark's configuration: no q / k norms, a sigmoid gate per head, three sliding-window layers
(512 positions, their caches rings of a window and a prefill chunk) per full-attention one, GELU's erf form.

- A dispatch runs up to 8 token slots through every layer and the LM head, streaming the weights once for all of them
  (`overlay.py`, `SparkOverlay`); the next step is patched into a second transaction while one runs.
- The prompt runs on a prefill (`prefill.py`, `SparkPrefill`): chunks of up to 2048 tokens as GEMMs on the same
  weights, fp32 residual norms and the attention of each layer kind, writing the caches and rings the decode
  continues from.
- DSpark (`dspark.py`, `DSparkOverlay`): the drafter's 5 layers run on the model's design and hardware context, its
  weights as BFP16 tiles in a buffer of their own, its head the first 64K rows (and 512 rows of the conversation's own
  tokens) of each column of the model's LM head, read in place. One dispatch a round takes the rows the model's last
  step kept (their layers' inputs read in place from the model's buffers) into the drafter's caches and runs the block
  of 7 rows that guesses the next 7 tokens; the host walks the block's Markov chain (`src/models/dspark.py`). One
  model step of 8 slots verifies the token and the 7 drafts; a draft is kept only where the model's own choice at its
  position is that draft, so greedy text is the same with and without the drafter. A prompt's chunks write the
  drafter's rows of its last window too.

Memory: the weights the NPU streams are re-laid-out into device buffers at load, in memory, at their quantization's
size; nothing is written to disk. The 4B's hidden size of 2560 is two and a half of the 1024-column tiles: the
feed-forward's gate / up and the LM head keep the last 512 columns of two rows' tiles in one tile, the attention's
inputs are padded to 3072 columns (the official 4B: a decode step streams 3.04 instead of 3.33 GB; the prompt's
matrix products skip the padding).

## Speed and accuracy (Strix Halo)

Prompts (`artlite-bench`, time to the first token, with the drafter's rows; essays):

| | 1K | 8K | 32K | 64K |
|---|---|---|---|---|
| 4B Q4_K_M | 0.462 s | 3.80 s | 19.9 s | 51.8 s |
| 4B bartowski Q4_K_M | 0.443 s | 3.67 s | 19.4 s | 50.7 s |
| 1.7B Q4_K_M | 0.209 s | 1.60 s | 7.48 s | 18.3 s |

Generation (greedy, 256 tokens; tokens/s without a drafter, then with DSpark):

| | essay summary, 1K | essay summary, 32K | code | math | JSON |
|---|---|---|---|---|---|
| 4B Q4_K_M | 17.3 -> 24.7 | 14.2 -> 20.4 | 16.6 -> 61.6 | 16.7 -> 56.5 | 16.7 -> 44.5 |
| 4B bartowski Q4_K_M | 18.6 -> 25.8 | 15.1 -> 22.0 | | | |
| 1.7B Q4_K_M | 39.5 -> 53.6 | 28.7 -> 36.5 | 38.1 -> 128.1 | 38.1 -> 134.5 | 38.1 -> 106.6 |

(code, math, JSON: short chat prompts, 320 tokens.) A round's drafts are kept 1.5 to 5 tokens at a time: text that the
drafter guesses (code, math, structured output) runs 2.7-3.7x as fast, prose 1.3-1.6x; where rounds keep too little the
generation falls back to steps of one token.

Accuracy:

| | |
|---|---|
| against the fp32 evaluation of the same file | teacher-forced KL, chat and essay: 4B 2.7e-3 / 1.7e-3, 1.7B 5.4e-3 / 1.8e-3 (llama.cpp's CPU build of the same file: 4B 2.4e-3 / 1.7e-3, 1.7B 3.9e-3 / 1.3e-3); a prompt through the prefill and the text after it 1.6-3.3e-3 |
| the drafter against XHToken's llama.cpp fork | tokens per round on the fork's six forced texts 3.61 vs 3.58 (1.7B); its first draft the fork-equivalent fp32 drafter's on 98-99.5% of positions |
| tools and math (4B, thinking off, greedy) | BFCL simple / multiple / parallel 124 / 150 (llama-server on the same file: 122); GSM8K's first 150: 124 (llama.cpp CPU: 124) |
| determinism | a token's logits depend on the tokens up to it only: steps of 1-8 tokens give the same logits bit for bit (greedy text with drafts is the text without) |

Design: the module docstrings of `src/models/spark/` and `src/artlite/decode/attn256.py`.
