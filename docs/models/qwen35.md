# Qwen3.5 / 3.6 / 3.8 and MiniCPM-V 4.7 on the NPU

Qwen3.6-27B, Qwen3.8-27B (without its MTP layer), Qwen3.5-9B and MiniCPM-V 4.7 1B (its language model, and images
through its vision encoder: [below](#images-minicpm-v-47)) from a GGUF file (llama.cpp architecture `qwen35`: Gated
DeltaNet layers with a full-attention layer every fourth), and Qwen3.6-35B-A3B (`qwen35moe`: the same layers with a
mixture of experts as the FFN: [below](#mixture-of-experts-qwen36-35b-a3b)). The architecture, tokenizer and chat template are read from the file's metadata.
Running them: [artlite-cli](../cli.md); `--think` lets the model think (default: an empty thought); default sampling:
the file's (`general.sampling.*`), else the model card's (MiniCPM-V's: temperature 0.7, top-p 0.95, top-k 100,
repetition penalty 1.03).

## Model files

```
$ARTLITE_MODELS_DIR/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf   (the main model)
$ARTLITE_MODELS_DIR/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf
$ARTLITE_MODELS_DIR/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf     (the development model)
$ARTLITE_MODELS_DIR/lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf
$ARTLITE_MODELS_DIR/ggml-org/Qwen3.6-35B-A3B-GGUF/mtp-Qwen3.6-35B-A3B-Q4_0.gguf          (its MTP drafter)
$ARTLITE_MODELS_DIR/openbmb/MiniCPM-V-4.7-1B-gguf/MiniCPM-V-4_7-1B-Q4_K_M.gguf     (MiniCPM-V's language model)
$ARTLITE_MODELS_DIR/openbmb/MiniCPM-V-4.7-1B-gguf/mmproj-model-f16.gguf            (its vision encoder, 16x)
$ARTLITE_MODELS_DIR/openbmb/MiniCPM-V-4.7-1B-gguf/mmproj-model-f16-4x.gguf         (the same, 4x: finer detail)
```

The models directory: [Installation](../install.md#models-directory). `model_config.qwen_gguf(quant, size)` names
the files (`size` "1B": MiniCPM-V's, also `model_config.minicpmv_gguf(quant)`; "35B-A3B"). The NPU path reads Q4_K,
Q5_K and Q6_K tensors (and an MoE router's F32 ones). A qwen35moe model's drafter (`model_config.qwen_mtp_of`): an
`mtp-*.gguf` next to the model, else in another account's repository of the same name (ggml-org publishes its drafters
apart from the models), Q4_0 first.

## How the NPU path works

Paths below are under `src/artlite/operators/`. One design (`qwen35/decode.py`) runs up to 4 tokens per dispatch
through every layer and the LM head; the host builds the dispatch's transaction (`qwen35/overlay.py`), and the cores
read what to do from their input streams. Per column:
- the main core runs the projections (the GEMV phases on the weight tiles, RMSNorm, the residual stream
  in fp32, SiLU(gate) up);
- the GDN core runs the Gated DeltaNet step: conv, gates, the delta rule on the fp32 state that it
  reads and writes back to DDR every token;
- the attention core runs the full attention (head_dim 256, partial RoPE, the sigmoid gate) and writes
  the position into the KV cache. The columns of a KV head (2 for the 9B and 27B, 4 for MiniCPM-V's 2 KV heads)
  share its cache from 96 positions on: each core reads a part, and the parts' results merge from core to core.

The prompt runs in chunks of up to 2048 tokens (the 35B-A3B: 4096, its experts' weights streamed once per chunk;
layer-major, in sub-chunks of 256; short prompts in chunks of 64 or 128) through the prefill (`qwen35/prefill.py`):
the prefill GEMMs, which read the same weight image, the Gated DeltaNet scan and the flash attention. A chunk's K / V,
GDN states and conv states land in the decode's buffers; a chunk runs as dispatches of whole layers, each within the
driver's hang detection. The prompt's last logits come from the LM head alone. Tokens where a chunk cannot start
(inside a 32-position cache group, e.g. after a generated reply), and a prompt's last 4 or fewer, run as decode steps.

Memory: the weights the NPU streams (15.95 GB of the 27B's file, 5.09 GB of the 9B's, 0.53 GB of MiniCPM-V's,
21.02 GB of the 35B-A3B's, and 0.56 GB of its drafter's) are re-laid-out into a device buffer at load, in memory, at
their quantization's size.
Nothing is written to disk. The prefill's buffers (a chunk's activations) take 0.12 GB (MiniCPM-V) to 0.97 GB (the
35B-A3B, 0.66 GB of them its experts' inputs and outputs).

## Mixture of experts (Qwen3.6-35B-A3B)

Each layer's FFN is 256 experts (8 per token) and a shared expert. A decode step computes the router's logits in fp32
on every column, then each column's cores pick every token's top 8 themselves and point the column's weight stream at
the experts they take (control packets the column writes and its shim reads back); each column computes whole experts
or halves of them, and the next phase adds the columns' results into the residual stream (`moe/route.py`,
`overlay/decode.py`). A step streams 1.96 GB of the image at one token (8 of the 256 experts in every layer).
A prompt's chunk routes its tokens on the host: each MoE layer's dispatch ends after the router, the host reads the
logits back, picks every token's top 8 (AVX-512) and writes each column's work list (chunks of up to 192 pairs of one
expert, dealt to the columns in turn) as control packets: the gather's while the shared expert runs, the experts'
GEMMs' and the combine's while the gather runs; the columns then gather their chunks' rows, run the experts' gate / up
and down on them (moving only the 64-row strips of a chunk that hold valid rows: most experts get a few dozen of a
chunk's 192), and the combine adds the routed and shared experts' outputs (`moe/prefill.py`).

## Multi-token prediction (Qwen3.6-35B-A3B)

With its drafter (the model's MTP layer: an attention layer and an MoE layer of its own, and an LM head), a generation
step takes up to three draft tokens along (`qwen35/draft.py`, `NpuQwen.generate`): the drafter's guesses of what
follows, one dispatch of the model's design each on the drafter's weights and KV cache (2.4-3.4 ms for 1-4 rows),
while the product of their probabilities stays at least 0.5. The step runs the token and its drafts, and a draft is
kept where the model's own token at its position is that draft; the GDN states stay uncommitted until the next step,
which replays the accepted tokens. The drafter's head is the first 65536 rows of its LM head (the most frequent
tokens: 98-99% of essay and code tokens). Its row at a position takes that position's token and the model's hidden
state of the position before, so its KV cache follows the model's: a prompt chunk also writes the drafter's records
of its rows (its eh_proj, attn_norm and K / V after the model's last layer: `qwen35/draft_prefill.py`), and the
accepted tokens' rows run with the next draft. The design runs the drafter's Q4_0 as a code page.

A step of several tokens sums the experts' outputs in another order than a step of one, so greedy text with drafts can
differ from greedy text without where the top two logits are within ~0.1 of each other.

## Images (MiniCPM-V 4.7)

```bash
artlite-cli -m openbmb/MiniCPM-V-4.7-1B-gguf/MiniCPM-V-4_7-1B-Q4_K_M.gguf \
    --mmproj openbmb/MiniCPM-V-4.7-1B-gguf/mmproj-model-f16.gguf --image photo.jpg -p "Describe the image."
```

`--mmproj` (artlite-cli and [artlite-server](../server.md), whose chat messages then take `image_url` parts) loads the
vision encoder. An image is cut as the official processor cuts it (transformers' `MiniCPMV4_6ImageProcessor`): an
overview of about 448 x 448 pixels and, for a larger image, a grid of up to `--max-slices` slices (default the file's 9;
the model card uses 36), each entry 64 tokens or a few more (the 4x file: 256). Its tokens take the official M-RoPE
positions (transformers' `get_rope_index`; llama.cpp numbers the markers like text, its PR #30510).

The encoder (`vision/minicpmv.py`) runs on Stable Diffusion's static design (`sd15/`, in a hardware context of its own):
one dispatch runs the whole encoder (27 layers, the ViT merger after layer 6, the final merger) on two entries, or on
one larger than 64 tokens alone, in about 60 ms: about 30 ms per entry (llama.cpp on the CPU: ~0.2 s per entry). The 4x
file has no ViT merger, so its 27 layers all run on 1024 tokens per entry: about 65 ms per entry. Its weights are
converted to bfp16 at the first load (the most sensitive ones, the patch embedding's and the final merger's, as bfp16
hi + lo pairs; the pixels enter as their exact 8-bit values) and cached with the dispatches, as Stable Diffusion's
(~0.7 GB, the cache directory; later loads ~0.3 s); an image's embeddings are kept, so a chat's next turns do not run
it again.

