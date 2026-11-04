# Qwen3 on the NPU

Qwen3 0.6B, its embedding and reranker models (Qwen3-Embedding-0.6B and -4B, Qwen3-Reranker-0.6B) and vllm-sr's
decision model Decision-2.0-Kai-0.6B (GGUF architectures `qwen3` and `vllm-decision2`) run on Llama's path: the same
layers, with q and k RMS-normed per head (a weight per dim) before RoPE. Running them: [artlite-cli](../cli.md),
[artlite-server](../server.md) (embeddings, rerank, `/classify`, `/v1/systemone`).

## Model files

```
$ARTLITE_MODELS_DIR/
  unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_1.gguf
  Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf
  Qwen/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q4_K_M.gguf
  ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf
  vllm-sr/Decision-2.0-Kai-0.6B/Decision-2.0-Kai-0.6B-Q8_0.gguf   (converted, below)
```

vllm-sr's own GGUF of Decision-2.0-Kai is a llama.cpp fork's format (`qwen3` with a `decision2` type) that neither
llama.cpp nor Artlite reads; llama.cpp's converter (its `requirements/requirements-convert_hf_to_gguf.txt`) makes the
`vllm-decision2` file from the checkpoint:

```bash
uvx --from huggingface_hub hf download vllm-sr/Decision-2.0-Kai-0.6B --local-dir ~/.artlite/models/vllm-sr/Decision-2.0-Kai-0.6B
python llama.cpp/convert_hf_to_gguf.py ~/.artlite/models/vllm-sr/Decision-2.0-Kai-0.6B --outtype q8_0 \
    --outfile ~/.artlite/models/vllm-sr/Decision-2.0-Kai-0.6B/Decision-2.0-Kai-0.6B-Q8_0.gguf
```

The tokenizer and the chat template are the GGUF file's (`models.tokenizer`, the `qwen2` pre-tokenizer; no BOS). The
chat template has Qwen3's thinking mode (`--think`); sampling follows the model card (thinking: temperature 0.6, top-k
20, top-p 0.95; else 0.7, 20, 0.8).

## On the NPU

- Decode runs on Llama's decode overlay (`models.llama.overlay`); its attention cores norm q and k with the layer's
  weights per dim (fp32, written next to the q and K / V objects at load) before RoPE.
- Prompts run through Llama's prefill (`models.llama.prefill`); its attention prep norms them likewise, and its Q
  panels are bfp16 hi + lo (the norm weights reach ~100: a single 8-bit block per 8 dims loses the small ones).
- The KV cache holds K as bfp16 hi + lo by default (`--kv-k hilo`; 96 KB per position): with one block per 8 dims
  (`--kv-k single`, 64 KB) the KL against the same file on the CPU goes from 0.0037 to 0.0085 on a chat.
- A file of Q8_0 alone (the embedding, reranker and decision files) takes 8-row tiles of 9216 bytes.
- Qwen3-Embedding-4B's hidden size (2560) and ffn (9728) are not whole K chunks of 1024: their projections run in
  chunks padded with zero weights.

Qwen3 0.6B Q4_1: time to first token at 1K / 8K / 32K prompt tokens 0.095 / 1.41 / 16.7 s; decode 9.8 ms per token
after 1K, 70.3 ms after 32K (101.5 / 14.2 tokens/s). Quality vs the CPU model on the same weights, teacher-forced: KL
0.0037 on a chat (through the prefill), 0.0037 after a 4800-token essay.

Through artlite-server, warm, each request's texts new: Qwen3-Embedding-0.6B Q8_0 a 30-token text's embedding in 31
ms, a 430-token passage's in 62 ms, batches of 8 passages 7.1K tokens/s; Qwen3-Embedding-4B Q4_K_M 87 ms, 229 ms, 1.9K
tokens/s; Qwen3-Reranker-0.6B Q8_0 16.3 pairs/s of ~510 tokens; Decision-2.0-Kai-0.6B 37-44 ms per question of 90-190
tokens.

Against llama.cpp on the same files (CPU): the embeddings within a cosine of 0.997-0.999 (4B: 0.994-0.999; its CPU and
Vulkan builds: 0.9995, 0.997; an exact fp32 model of the 0.6B file: 0.999 on its most sensitive texts, llama.cpp's own
8-bit activations), the same share of target passages among the 30 test queries' top 10 (72%; 4B 73%, CPU 74%); the
rerank scores within 0.18 (logits within 1.2), the target passage first as often (93%); Kai's answers 314 of 320 the
same, probabilities within 0.07 (median 0.006), accuracy on four classification sets within one question of 50. The
prefill's GEMMs take bfp16 weights and activations (8-bit mantissas per block of 8), so a row is within ~0.15% of the
CPU's where llama.cpp's own builds differ by 0.05%; these models' attention takes its softmax weights as two blocks
each (hi + lo: its error is then its output's own rounding).

NPU code: `src/models/llama/` (the model object `Qwen3`: `qwen3.py`).
