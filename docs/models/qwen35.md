# Qwen3.5 / 3.6 / 3.8 on the NPU

Qwen3.6-27B, Qwen3.8-27B (without its MTP layer) and Qwen3.5-9B from a GGUF file (llama.cpp architecture `qwen35`:
Gated DeltaNet layers with a full-attention layer every fourth). The architecture, tokenizer and chat template are
read from the file's metadata. Running them: [artlite-cli](../cli.md); `--think` lets the model think (default: an
empty thought); default sampling: the file's (`general.sampling.*`), else the model card's.

## Model files

```
$ARTLITE_MODELS_DIR/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf   (the main model)
$ARTLITE_MODELS_DIR/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf
$ARTLITE_MODELS_DIR/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf     (the development model)
```

The models directory: [Installation](../install.md#models-directory). `model_config.qwen_gguf(quant, size)` names
the files. The NPU path reads Q4_K, Q5_K and Q6_K tensors.

## How the NPU path works

Paths below are under `src/artlite/operators/`. One design (`qwen35/decode.py`) runs up to 4 tokens per dispatch
through every layer and the LM head; the host builds the dispatch's transaction (`qwen35/overlay.py`), and the cores
read what to do from their input streams. Per column:
- the main core runs the projections (the GEMV phases on the weight tiles, RMSNorm, the residual stream
  in fp32, SiLU(gate) up);
- the GDN core runs the Gated DeltaNet step: conv, gates, the delta rule on the fp32 state that it
  reads and writes back to DDR every token;
- the attention core runs the full attention (head_dim 256, partial RoPE, the sigmoid gate) and writes
  the position into the KV cache.

The prompt runs in chunks of up to 2048 tokens (layer-major, in sub-chunks of 256) through the prefill
(`qwen35/prefill.py`): the prefill GEMMs, which read the same weight image, the Gated DeltaNet scan
and the flash attention. A chunk's K / V, GDN states and conv states land in the decode's buffers; a chunk
runs as dispatches of whole layers, each within the driver's hang detection. The prompt's last logits come
from the LM head alone. Tokens where a chunk cannot start (inside a 32-position cache group, e.g. after a
generated reply; fewer than 3 left) run as decode steps.

Memory: the weights the NPU streams (15.95 GB of the 27B's file, 5.09 GB of the 9B's) are
re-laid-out into a device buffer at load, in memory, at their quantization's size. Nothing is written
to disk.
