# Llama 3 on the NPU

Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct (the shapes and the llama3 RoPE scaling come
from the GGUF file's metadata), GGUF files of Q4_K_M / Q4_K_S, Q5_K_M, Q6_K, Q4_0, Q4_1, Q3_K_M / Q3_K_L, Q2_K and
BF16 weights (BF16 tensors as bfp16 tiles). Running them: [artlite-cli](../cli.md).

## Model files

```
$ARTLITE_MODELS_DIR/lmstudio-community/
  Llama-3.2-1B-Instruct-GGUF/     unsloth/Llama-3.2-1B-Instruct-GGUF
    Llama-3.2-1B-Instruct-Q4_K_M.gguf, ...
  Llama-3.2-1B-Instruct/          unsloth/Llama-3.2-1B-Instruct (= meta-llama, ungated; optional)
    tokenizer.json                (test_llama_tokenizer compares the GGUF vocabulary's ids with it)
  Llama-3.2-3B-Instruct-GGUF/     unsloth/Llama-3.2-3B-Instruct-GGUF
    Llama-3.2-3B-Instruct-Q4_K_M.gguf, ...
  Meta-Llama-3.1-8B-Instruct-GGUF/  lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF
    Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

The models directory: [Installation](../install.md#models-directory). Other uploaders' quantizations work the same
way (`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/...-Q4_K_S.gguf`,
`MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct.Q5_K_M.gguf`).

```bash
uvx --from huggingface_hub hf download unsloth/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --local-dir ~/.artlite/models/lmstudio-community/Llama-3.2-1B-Instruct-GGUF
```

The tokenizer (gigatoken) is the GGUF file's own vocabulary, or the `tokenizer.json` next to
the weights; both give the same token ids for Llama 3. The chat template is the GGUF file's.

## On the NPU

- Decode runs on the shared decode overlay (`artlite.operators.llama.overlay`), up to 4 tokens per dispatch. The
  weights are re-laid-out at load into a device buffer at their quantization's size; nothing is written to disk.
- Prompts run through the prefill (`artlite.operators.llama.prefill`): chunks of up to 4096 tokens (8B: 2048), layer-major
  programs assembled from one library ELF (GEMMs on the decode's weight image, the bf16 residual stream, a flash
  attention writing the decode's KV caches); short prompts in chunks of 64 / 128 tokens (1B: 128). Tokens where a chunk cannot
  start (inside a cache group: after a reply) run as decode steps of up to 4 tokens.
- The KV cache (`-c`, up to the model's 131072 positions) holds K and V in bfp16ebs8: 18 KB per position for 1B,
  64.5 / 73.7 KB for 3B / 8B (2.4 / 8.5 / 9.7 GB at 128K). `--kv-k hilo` stores K as bfp16 hi + lo blocks: 1.5x the
  cache and its read time at long contexts, with no measured quality gain for Llama.

Time to first token at 1K / 8K / 32K prompt tokens: 1B Q4_0 0.109 / 1.07 / 8.59 s, 3B Q4_K_M 0.305 / 2.78 / 19.1 s,
8B Q4_K_M 0.653 / 6.16 / 40.9 s; a 48-token chat prompt: 1B Q4_0 35 ms, 3B 70 ms, 8B 129 ms. Decode per token at 100 /
32K positions (Q4_K_M): 1B 14.8 / 25.6 ms, 3B 36.7 / 74.2 ms, 8B 83.3 / 125.8 ms (hi + lo K at 32K: 30.7 / 92.6 /
147.2 ms); at 100 positions 1B Q4_0 14.1, Q4_1 15.1, Q3_K_L 14.2, BF16 25.0 ms, 3B Q2_K 26.3, Q3_K_M 31.9 ms. Quality vs the CPU model
on the same weights (three chat prompts and three essay windows through the prefill, their continuations
teacher-forced): KL 0.0016 / 0.0011 / 0.0011 for 1B / 3B / 8B.

NPU code: `src/models/llama3/llama.py` (the model objects `Llama` and `NpuLlama`).
