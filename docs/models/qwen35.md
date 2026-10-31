# Qwen3.5 / 3.6 / 3.8 and MiniCPM-V 4.7 on the NPU

Qwen3.6-27B, Qwen3.8-27B (with its MTP layer as the drafter), Qwen3.5-9B and MiniCPM-V 4.7 1B (its language model, and images
through its vision encoder: [below](#images-minicpm-v-47)) from a GGUF file (llama.cpp architecture `qwen35`: Gated
DeltaNet layers with a full-attention layer every fourth), and Qwen3.6-35B-A3B (`qwen35moe`: the same layers with a
mixture of experts as the FFN: [below](#mixture-of-experts-qwen36-35b-a3b)), also as Occamy 1.0 (Accio-Lab's fine-tune
of it for agentic tool use, abliterated by IsValorum, in a 3-bit quantization: 15.8 GB instead of 21 GB). The
architecture, tokenizer and chat template are read from the file's metadata.
Running them: [artlite-cli](../cli.md); `--think` lets the model think (default: an empty thought); default sampling:
the file's (`general.sampling.*`), else the model card's (MiniCPM-V's: temperature 0.7, top-p 0.95, top-k 100,
repetition penalty 1.03).

## Model files

```
$ARTLITE_MODELS_DIR/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf   (the main model)
$ARTLITE_MODELS_DIR/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf   (with its MTP layer)
$ARTLITE_MODELS_DIR/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf     (the development model)
$ARTLITE_MODELS_DIR/lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf
$ARTLITE_MODELS_DIR/ggml-org/Qwen3.6-35B-A3B-GGUF/mtp-Qwen3.6-35B-A3B-Q4_0.gguf          (its MTP drafter)
$ARTLITE_MODELS_DIR/IsValorum/Occamy-1.0-APEX-I-MiniPlus-V2.1-Abliterated-GGUF/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf
$ARTLITE_MODELS_DIR/IsValorum/Occamy-1.0-APEX-I-MiniPlus-V2.1-Abliterated-GGUF/occamy-1.0-mtp-Q4_K_M.gguf (its drafter)
$ARTLITE_MODELS_DIR/openbmb/MiniCPM-V-4.7-1B-gguf/MiniCPM-V-4_7-1B-Q4_K_M.gguf     (MiniCPM-V's language model)
$ARTLITE_MODELS_DIR/openbmb/MiniCPM-V-4.7-1B-gguf/mmproj-model-f16.gguf            (its vision encoder, 16x)
$ARTLITE_MODELS_DIR/openbmb/MiniCPM-V-4.7-1B-gguf/mmproj-model-f16-4x.gguf         (the same, 4x: finer detail)
```

The models directory: [Installation](../install.md#models-directory). `model_config.qwen_gguf(quant, size)` names
the files (`size` "1B": MiniCPM-V's, also `model_config.minicpmv_gguf(quant)`; "35B-A3B"; Occamy's:
`model_config.occamy_gguf()`). The NPU path reads Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 and IQ3_XXS tensors, Q4_0 ones (a
drafter's) and F32 ones (an MoE router's, Occamy's GDN alpha). Qwen3.8's drafter is its file's own MTP layer; a
qwen35moe model's (`model_config.qwen_mtp_of`):
an `mtp-*.gguf` (or an uploader's `<name>-mtp-*.gguf`) next to the model, else in another account's repository of the
same name (ggml-org publishes its drafters apart from the models); of several, Q4_0, then Q4_K_M, then Q8_0.

## How the NPU path works

Paths below are under `src/models/` (the model's own code: `qwen35/`, `minicpmv/`) and `src/artlite/` (what the models
share). One design (`qwen35/decode.py`) runs up to 4 tokens per dispatch through every layer and the LM head (the 27B
models' up to 8, for a drafter's long chains: a step of 8 costs 1.01x a step of one at 1K and at 32K); the host builds the dispatch's transaction (`qwen35/overlay.py`), and the
cores read what to do from their input streams. Per column:
- the main core runs the projections (the GEMV phases on the weight tiles, RMSNorm, the residual stream
  in fp32, SiLU(gate) up);
- the GDN core runs the Gated DeltaNet step: conv, gates, the delta rule on the fp32 state that it
  reads and writes back to DDR every token (in a design of 8 tokens and on the MoE models chunkwise, one pass over the state
  for a step's tokens, as the prefill; in a design of 8 tokens the main core's helper, idle then, computes the heads' constants);
- the attention core runs the full attention (head_dim 256, partial RoPE, the sigmoid gate) and writes
  the position into the KV cache. The columns of a KV head (2 for the 9B and 27B, 4 for MiniCPM-V's 2 KV heads)
  share its cache from 96 positions on: each core reads a part, and the parts' results merge from core to core (in a design of 8 tokens one stream of the cache reaches both cores, each attending with its own heads: one pass over the cache per step; on the 35B-A3B and Occamy steps of 3 and 4 tokens split a chain of 4 into two halves on shared streams, also one pass).

The prompt runs in chunks of up to 2048 tokens (the 35B-A3B: 4096, its experts' weights streamed once per chunk;
layer-major, in sub-chunks of 256; short prompts in chunks of 64 or 128) through the prefill (`qwen35/prefill.py`):
the prefill GEMMs, which read the same weight image, the Gated DeltaNet scan and the flash attention. A chunk's K / V,
GDN states and conv states land in the decode's buffers; a chunk runs as dispatches of whole layers, each within the
driver's hang detection. The prompt's last logits come from the LM head alone. Tokens where a chunk cannot start
(inside a 32-position cache group, e.g. after a generated reply), and a prompt's last 4 or fewer, run as decode steps.

Memory: the weights the NPU streams (15.95 GB of a 27B's file, and 0.27 GB of Qwen3.8's MTP layer; 5.09 GB of the
9B's, 0.53 GB of MiniCPM-V's, 21.02 GB of the 35B-A3B's, and 0.56 GB of its drafter's; 15.82 GB of Occamy's, and
0.66 GB of its drafter's) are re-laid-out into a device buffer at load, in memory, at their quantization's size.
Nothing is written to disk. The prefill's buffers (a chunk's activations) take 0.12 GB (MiniCPM-V) to 0.97 GB (the
35B-A3B, 0.66 GB of them its experts' inputs and outputs).

## Mixture of experts (Qwen3.6-35B-A3B)

Each layer's FFN is 256 experts (8 per token) and a shared expert. A decode step computes the router's logits in fp32
on every column, then each column's cores pick every token's top 8 themselves and point the column's weight stream at
the experts they take (control packets the column writes and its shim reads back); each column computes whole experts
or halves of them, and the next phase adds the columns' results into the residual stream (`moe/route.py`,
`decode/design.py`). A step streams 1.96 GB of the image at one token (8 of the 256 experts in every layer; Occamy's
file 1.79 GB).
A prompt's chunk routes its tokens on the host: each MoE layer's dispatch ends after the router, the host reads the
logits back, picks every token's top 8 (AVX-512) and writes each column's work list (chunks of up to 192 pairs of one
expert, dealt to the columns in turn) as control packets: the gather's while the shared expert runs, the experts'
GEMMs' and the combine's while the gather runs; the columns then gather their chunks' rows, run the experts' gate / up
and down on them (moving only the 64-row strips of a chunk that hold valid rows: most experts get a few dozen of a
chunk's 192), and the combine adds the routed and shared experts' outputs (`moe/prefill.py`).

Occamy's file (APEX-I-MiniPlus) keeps the experts of the first and last ten layers in Q3_K and those of the middle
twenty in IQ3_XXS, the shared experts in Q5_K, the GDN z in Q8_0 and alpha in F32. Its 9280-byte tiles hold 21 Q3_K
rows or 23 IQ3_XXS rows (the 35B-A3B's at most 16), and its design loads each format's kernel as a code page when a tile
needs it. IQ3_XXS stores indices into a 256-entry grid of 4 values each: the cores look them up by gathers from copies
of the grid in their own memory (`formats/iq3.py`): in a decode step per 8 x 8 weight panel, in a prompt's GEMMs per
super-block of the rows a pass dequantizes. Alpha runs as two BFP16 planes (hi + lo) summed in fp32.

## Multi-token prediction (Qwen3.8-27B, Qwen3.6-35B-A3B, Occamy 1.0)

With its drafter (the model's MTP layer: an attention layer and an FFN of its own - Qwen3.8-27B's dense, in its model
file; the 35B-A3B's an MoE layer, in a file of its own, with its own LM head), a generation step takes up to three draft
tokens along (Qwen3.8-27B up to seven; `qwen35/draft.py`, `OverlayModel.generate`): the drafter's guesses of what
follows, one dispatch of the model's design each on the drafter's weights and KV cache (2.4-3.4 ms for 1-4 rows with
ggml-org's Q4_0 drafter, 3.2-4.7 ms with Occamy's Q4_K_M one, ~11 ms with Qwen3.8-27B's: 14 ms at 32K), as long as their
chain's probability pays for what they cost: a draft's slot in the step and the drafter's step for the next one, against
the tokens per second the model's rounds give (`OverlayModel._worth`; a dense model's slot costs ~1.5% of its step, so
its chains go on down to a probability of ~0.1, an MoE's slots route to more experts: ~0.4-0.5). The step runs the token
and its drafts, and a draft is kept where the model's own token at its position is that draft; the GDN states stay
uncommitted until the next step, which replays the accepted tokens. The drafter's head is about 66048 rows of its LM
head: the special tokens (the turn's end, tool calls, a thought's end), the first ids (the most frequent tokens) and 512
of the sequence's own tokens past them, the most recently seen (an agent's turns repeat the identifiers of the code it
reads: 0.2% of their tokens are outside the head, 2.7% outside the first 65536 ids). Qwen3.8-27B's MTP layer shares the
model's LM head: the model's head rows put those first in each column (67584 rows: whole tiles), so the drafter streams
the first rows of the model's own head, and the model's logits are gathered back into token order. Its row at a position
takes that position's token and the model's hidden state of the position before, so its KV cache follows the model's: a
prompt chunk also writes the drafter's records of its rows (its eh_proj, attn_norm and K / V after the model's last
layer: `qwen35/draft_prefill.py`), and the accepted tokens' rows run with the next draft. The design runs the drafter's
formats as code pages.

On the 35B-A3B models a step of several tokens sums the experts' outputs in another order than a step of one, so
greedy text with drafts can differ from greedy text without where the top two logits are within ~0.1 of each other; the
dense models' steps give the numbers of steps of one, so their greedy text is the same with drafts or without.

Every qwen35 model, with a drafter or not, also takes draft tokens from the prompt where the text repeats it
(`models/drafts.py`, [artlite-cli](../cli.md#text)); a step of them leaves the GDN states uncommitted as a drafter's
does, and with a drafter its rows follow the step's kept tokens.

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

The encoder (`minicpmv/encoder.py`) runs on the vision overlay, Stable Diffusion's static design (`vision/`, in a
hardware context of its own): one dispatch runs the whole encoder (27 layers, the ViT merger after layer 6, the final
merger) on two entries, or on one larger than 64 tokens alone, in about 60 ms: about 30 ms per entry (llama.cpp on the
CPU: ~0.2 s per entry). The 4x file has no ViT merger, so its 27 layers all run on 1024 tokens per entry: about 65 ms
per entry. Its weights are converted to bfp16 at the first load (the most sensitive ones, the patch embedding's and the
final merger's, as bfp16 hi + lo pairs; the pixels enter as their exact 8-bit values) and cached with the dispatches, as
Stable Diffusion's (~0.7 GB, the cache directory; later loads ~0.3 s); an image's embeddings are kept, so a chat's next
turns do not run it again.

