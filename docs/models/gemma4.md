# Gemma 4 on the NPU

Gemma 4 E4B from a GGUF file. The architecture is read from the file's metadata (no
`config.json`), the tokenizer and the chat template from the file itself. Running it: [artlite-cli](../cli.md);
default sampling: the model's (temperature 1.0, top-k 64, top-p 0.95).

## Model files

```
$ARTLITE_MODELS_DIR/ggml-org/gemma-4-E4B-it-GGUF/     ggml-org/gemma-4-E4B-it-GGUF
    gemma-4-E4B-it-Q4_0.gguf                          (the NPU path: Q4_0 layers, Q8_0 embedding)
    mtp-gemma-4-E4B-it-Q4_0.gguf                      (the model's drafter, gemma4-assistant: multi-token prediction)
    gemma-4-E4B-it-Q8_0.gguf, gemma-4-E4B-it-BF16.gguf   (the fp32 reference only, for now: tests/gemma4/cpu.py)
```

The models directory: [Installation](../install.md#models-directory).

## How the NPU path works

Paths below are under `src/models/` (Gemma's own code: `gemma4/`) and `src/artlite/` (what the models share). One
design, compiled once: the decode overlay shared with Qwen3.5 and Llama (`artlite.decode`), with the attention cores it
shares with Llama in Gemma's configuration (`decode/attn.py`, `gemma4/attn.py`). The host builds every dispatch's
transaction (`gemma4/overlay.py`, `GemmaOverlay`); the cores read what to do from the metadata of their input objects.
The host reads the tokens' embedding rows and per-layer embedding rows (`src/models/gemma4/host.py`); a dispatch's
first phase projects the embeddings to the per-layer inputs (the host keeps a token's projection: a dispatch of kept
tokens writes them instead; the prefill's chunks: on the host).
- A dispatch runs up to 4 token slots through all 42 layers and the LM head, streaming the weights
  once for all of them.
- Cores per column: the main core and its helper run the projections (Q4_0 / Q8_0 tiles on the
  MMUL), the attention core runs the attention of its KV head's chain and the KV cache.
- The prompt runs on a prefill (`gemma4/prefill.py`, `GemmaPrefill`): chunks of up to 2016 tokens
  through the 24 layers that own a KV cache (the others only read those caches), as GEMMs on the
  same weights (`prefill/gemm.py`), norms, attention and per-layer-input operators,
  writing the caches the decode continues from. A dispatch from layer 23 on (the layers after the
  prefill's, the LM head) finishes the prompt's last token.
- Generation with the model's drafter (`mtp-<model file>` next to it; `gemma4/draft.py`): a draft step
  (~1.2 ms, a dispatch of the same design on the drafter's weights and the model's caches) guesses the
  next three tokens; one dispatch of four slots verifies them, and a draft is kept only where the
  model's own choice at its position is that draft.

Memory:
- The weights the NPU streams (3.04 GB of the 4.59 GB file, the per-layer projection as a Q8_0 copy among them) are
  re-laid-out into a device buffer at load, in memory, at their quantization's size. Nothing is written to disk.
- The per-layer token embedding (1.6 GB) stays in the mapped file: one row per token is read.
- The KV cache: 222 MB at 4096 positions (every cache linear in the positions, the sliding-window
  ones too: the prefill writes and reads them as the decode does).

Speed and accuracy (E4B Q4_0, Strix):

| | |
|---|---|
| generation | 17.4 tokens/s at a short context, 16.5 at 600 positions (a dispatch of one token 56-59 ms; streaming the 3.0 GB of weights and the inputs at the NPU's DDR read ceiling takes 54.6 ms) |
| prompt | 2345 tokens/s at 1032 tokens, 2750 at 8000 (`--no-prefill`: decode steps of 4 tokens, 121) |
| accuracy vs the fp32 evaluation of the same file | teacher-forced KL 0.00023 / 0.00026 at 1 / 4 tokens per dispatch (34 tokens, argmax agreement 100%) |
| after a prompt through the prefill | KL 0.00020 (34 tokens); its KV caches within 1-3% of the decode path's at every position through layer 11 (deeper layers amplify the prefill's rounding at a few positions) |
| determinism | a token's logits depend on the tokens up to it only: not on the other slots of its dispatch, nor on rejected drafts; steps of 1-4 tokens give the same logits bit for bit (greedy text with drafts is the text without) |

Design: the module docstrings of `decode/design.py` and `gemma4/`.
