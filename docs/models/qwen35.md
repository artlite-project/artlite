# Qwen3.5 / 3.6 / 3.8 and MiniCPM-V 4.7 on the NPU

Qwen3.6-27B, Qwen3.8-27B (with its MTP layer as the drafter), Qwen3.5-9B and MiniCPM-V 4.7 1B (its language model, and images
through its vision encoder: [below](#images-minicpm-v-47)) from a GGUF file (llama.cpp architecture `qwen35`: Gated
DeltaNet layers with a full-attention layer every fourth), and Qwen3.6-35B-A3B (`qwen35moe`: the same layers with a
mixture of experts as the FFN: [below](#mixture-of-experts-qwen36-35b-a3b)), also as Occamy 1.0 (Accio-Lab's fine-tune
of it for agentic tool use, abliterated by IsValorum, in a 3-bit quantization: 15.8 GB instead of 21 GB), and the
decision models vllm-sr d3-lite, d3-nano and Kev-4B ([below](#decision-models-vllm-d3-kev)); the Qwen models take images
through their Qwen3-VL vision encoder ([below](#images-qwen35--36--38)). The architecture, tokenizer and chat template
are read from the file's metadata.
Running them: [artlite-cli](../cli.md); `--think` lets the model think (default: an empty thought); default sampling:
the file's (`general.sampling.*`), else the model card's (MiniCPM-V's: temperature 0.7, top-p 0.95, top-k 100,
repetition penalty 1.03).

## Model files

```
$ARTLITE_MODELS_DIR/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf   (the main model)
$ARTLITE_MODELS_DIR/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf   (with its MTP layer)
$ARTLITE_MODELS_DIR/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf           (unsloth's dynamic quantization, with its MTP layer)
$ARTLITE_MODELS_DIR/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf          (smaller ones, with IQ1_S and IQ1_M tensors;
$ARTLITE_MODELS_DIR/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ1_M.gguf             UD-IQ1_M without an MTP layer)
$ARTLITE_MODELS_DIR/z-lab/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf   (its DFlash2 drafter, optional, the default where it is here)
$ARTLITE_MODELS_DIR/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf     (the development model)
$ARTLITE_MODELS_DIR/lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf
$ARTLITE_MODELS_DIR/lmstudio-community/Qwen3.6-35B-A3B-GGUF/mmproj-Qwen3.6-35B-A3B-BF16.gguf  (its vision encoder; the 27B models' likewise)
$ARTLITE_MODELS_DIR/ggml-org/Qwen3.6-35B-A3B-GGUF/mtp-Qwen3.6-35B-A3B-Q4_0.gguf          (its MTP drafter)
$ARTLITE_MODELS_DIR/IsValorum/Occamy-1.0-APEX-I-MiniPlus-V2.1-Abliterated-GGUF/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf
$ARTLITE_MODELS_DIR/IsValorum/Occamy-1.0-APEX-I-MiniPlus-V2.1-Abliterated-GGUF/occamy-1.0-mtp-Q4_K_M.gguf (its drafter)
$ARTLITE_MODELS_DIR/IsValorum/Occamy-1.0-APEX-I-MiniPlus-V2.1-Abliterated-GGUF/mmproj-Q8_0.gguf (its vision encoder)
$ARTLITE_MODELS_DIR/openbmb/MiniCPM-V-4.7-1B-gguf/MiniCPM-V-4_7-1B-Q4_K_M.gguf     (MiniCPM-V's language model)
$ARTLITE_MODELS_DIR/openbmb/MiniCPM-V-4.7-1B-gguf/mmproj-model-f16.gguf            (its vision encoder, 16x)
$ARTLITE_MODELS_DIR/openbmb/MiniCPM-V-4.7-1B-gguf/mmproj-model-f16-4x.gguf         (the same, 4x: finer detail)
$ARTLITE_MODELS_DIR/ggml-org/vllm-d3-lite-GGUF/vllm-d3-lite-Q8_0.gguf              (decision models: artlite-server)
$ARTLITE_MODELS_DIR/ggml-org/vllm-d3-lite-GGUF/mmproj-vllm-d3-lite-Q8_0.gguf       (its vision encoder: pictures in requests)
$ARTLITE_MODELS_DIR/ggml-org/vllm-d3-nano-GGUF/vllm-d3-nano-Q8_0.gguf
$ARTLITE_MODELS_DIR/ggml-org/vllm-d3-nano-GGUF/mmproj-vllm-d3-nano-Q8_0.gguf
$ARTLITE_MODELS_DIR/ggml-org/Kev-4B-GGUF/Kev-4B-Q4_K_M.gguf
```

The models directory: [Installation](../install.md#models-directory). `model_config.qwen_gguf(quant, size)` names
the files (`size` "1B": MiniCPM-V's, also `model_config.minicpmv_gguf(quant)`; "35B-A3B"; Occamy's:
`model_config.occamy_gguf()`; Swift 1.5's: `model_config.swift_gguf()`; unsloth's: `model_config.unsloth_qwen_gguf(quant)`).
The NPU path reads Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS and
IQ4_NL tensors, Q4_0 and Q4_1 ones and F32 ones (an MoE router's, a GDN alpha), in any role, and a dense layer's FFN
gate and up in two of them (unsloth's dynamic quantizations: below). A dense model with a KV head per 2 columns decodes
up to 8 tokens per dispatch where its file holds Q4_K and Q6_K tensors only, a drafter runs or the file has more than
ten formats; else up to 4 (`overlay.slots_of`). Qwen3.8's drafter is its DFlash2 file (`model_config.qwen_dflash_of`:
the Q4_K_M file of a `<repository>-DFlash2-GGUF` of any account; a PARO checkpoint's: its base model's) where the
design runs 8 tokens and the file has its MTP layer (the DFlash2 drafter's head), or the model is a PARO checkpoint (its
design of 4 slots), else its file's own MTP layer (unsloth's UD-IQ1_M has none: no drafter); a qwen35moe
model's (`model_config.qwen_mtp_of`): an `mtp-*.gguf` (or an uploader's `<name>-mtp-*.gguf`) next to the model, else in
another account's repository of the same name (ggml-org publishes its drafters apart from the models); of several, Q4_0,
then Q4_K_M, then Q8_0.
`--spec-draft-model FILE` takes another file, `--spec-type none` none ([artlite-cli](../cli.md)).

## How the NPU path works

Paths below are under `src/models/` (the model's own code: `qwen35/`, `minicpmv/`) and `src/artlite/` (what the models
share). One design (`qwen35/decode.py`) runs up to 4 tokens per dispatch through every layer and the LM head (the 27B
models' up to 8 when their gate / up is Q4_K, for a drafter's long chains: a step of 8 costs 1.01x a step of one at 1K and at 32K); the host builds the dispatch's transaction (`qwen35/overlay.py`), and the
cores read what to do from their input streams. Per column:
- the main core runs the projections (the GEMV phases on the weight tiles, RMSNorm, the residual stream
  in fp32, SiLU(gate) up);
- the GDN core runs the Gated DeltaNet step: conv, gates, the delta rule on the fp32 state that it
  reads and writes back to DDR every token (in a design of 8 tokens and on the MoE models chunkwise, one pass over the state
  for a step's tokens, as the prefill, the state written back only by the step that ends a block of 8 positions; in a design
  of 8 tokens the main core's helper, idle then, computes the heads' constants);
- the attention core runs the full attention (head_dim 256, partial RoPE, the sigmoid gate) and writes
  the position into the KV cache. The columns of a KV head (2 for the 9B and 27B, 4 for MiniCPM-V's 2 KV heads)
  share its cache from 96 positions on: each core reads a part, and the parts' results merge from core to core (in a design of 8 tokens one stream of the cache reaches both cores, each attending with its own heads: one pass over the cache per step; on the 35B-A3B and Occamy steps of 3 and 4 tokens split a chain of 4 into two halves on shared streams, also one pass).

The LM head is screened where it is Q6_K and the design has room for its tiles (`artlite/decode/screen.py`; not the 27B
models with an MTP layer: their design of 8 tokens has none, and their drafters read the head's rows in place): the NPU
reads only the head's scales and the top 2 bits of its codes (39% of its bytes: 7 - 40% of what these models stream per
token is their head), which gives every token's logit to within ~0.7. The host then computes the logits that can still
be among the largest exactly, from the head's rows in the model file (a few rows for the most likely token, ~100 for a
sampler's 20 candidates, 0.2 - 0.3 ms): the choices are those of the exact fp32 head (the cores' own bf16 logits differ
from it in the most likely token of ~0.5% of the steps). `ARTLITE_LM_SCREEN=0` reads the whole head on the NPU.

The prompt runs in chunks of up to 2048 tokens (the 35B-A3B: 4096, its experts' weights streamed once per chunk;
MiniCPM-V: 2304, its full attention split between pairs of cores, `artlite/prefill/attn256_split.py`, which stream
the cache two thirds as often; layer-major, in sub-chunks of 256; short prompts in chunks of 64 or 128) through the
prefill (`qwen35/prefill.py`):
the prefill GEMMs, which read the same weight image, the Gated DeltaNet scan and the flash attention. A chunk's K / V,
GDN states and conv states land in the decode's buffers; a chunk runs as dispatches of whole layers, each within the
driver's hang detection. The prompt's last logits come from the LM head alone. Tokens where a chunk cannot start
(inside a 32-position cache group, e.g. after a generated reply), and a prompt's last 4 or fewer, run as decode steps.

Several requests at a time (`artlite-server -np N`, up to 8; MiniCPM-V's language model, Qwen3.5-9B, Qwen3.6-27B,
Qwen3.8-27B and Swift 1.5: the dense files, on the design of 8 tokens per dispatch): the NPU holds N sequences' KV caches and Gated DeltaNet states (0.1 GB a sequence on the 9B, 0.3 GB on the 27B), and a decode
step's 8 slots are tokens of the requests that generate (`models/lanes.py`): the weights stream once for all of them,
each attention layer runs a pass per sequence over that sequence's cache and each Gated DeltaNet layer a chain per
sequence on its state (`qwen35/gdn_steps.py`). A step of 2 / 4 / 8 requests takes 1.05x / 1.12x / 1.39x a step of one,
so 4 requests get 3.2x and 8 get 4.8x one request's tokens per second in all (the 27B; the 9B: 1.07x / 1.16x / 1.50x,
3.3x and 4.8x). The LM head is then read whole, and the 9B, whose three formats' kernels do not fit a main core of
that design together, loads each as a code page: one request alone gets 10.7 tokens per second there instead of
12.5 (MiniCPM-V, whose own design is of 4 tokens too: 98 instead of 115, and 135 / 164 / 172 for 2 / 4 / 8 requests).
A file's MTP layer (Qwen3.8-27B's, Swift 1.5's) drafts for every request (a cache of its own per sequence, its guesses for a request ahead
of the step the requests share, for that request's share of the step's slots: 8.9 / 13.5 / 18.0 tokens per second
for 1 / 2 / 4 requests); the DFlash2 drafter, whose records follow one sequence, does not load then:
[artlite-server](../server.md#several-requests-at-a-time). The mixture-of-experts models serve `-np N` one request at a time.

Memory: the weights the NPU streams (15.95 GB of a 27B's file, and 0.27 GB of Qwen3.8's MTP layer; 4.60 GB of the
9B's, 0.41 GB of MiniCPM-V's, 20.77 GB of the 35B-A3B's, and 0.56 GB of its drafter's; 15.82 GB of Occamy's, and
0.66 GB of its drafter's) are re-laid-out into a device buffer at load, in memory, at their quantization's size (of a
screened LM head only the part the NPU reads, 41% of it: the host reads the rest from the file).
Nothing is written to disk. The prefill's buffers (a chunk's activations) take 0.12 GB (MiniCPM-V) to 0.97 GB (the
35B-A3B, 0.66 GB of them its experts' inputs and outputs).

## Decision models (vllm-d3, kev)

vllm-sr's d3-lite (Qwen3.5-0.8B's shapes) and d3-nano (2B) answer questions about a state with the logits of the
options' label tokens at a prompt's last token ([artlite-server's `/v1/systemone`](../server.md#decisions); no text
generation). Their full-attention layers are non-causal (`qwen35.attention.causal` false): a prompt runs whole in one
prefill chunk (up to 2048 tokens) whose attention passes see all of its tokens, the padding after them masked; the Gated
DeltaNet layers stay recurrent. Every tensor of ggml-org's files is Q8_0 (8-row tiles; gate / up in passes of half a
group; d3-lite's down, of its ffn 3584, reads 3.5 of a row's 4 tiles), and the weights go to the NPU without the LM head
(an option label's logit is the host's: 0.59 instead of 0.87 GB, d3-nano 1.56 instead of 2.14). On 320 questions (SST-2,
AG News, BoolQ and MNLI rows, questions about essay passages) their choices are llama.cpp's (CPU) on 318 and their
accuracy the same within one question of 50, the probabilities within 0.08 of its (median 0.002-0.003: the prefill's
bfp16 GEMMs; llama.cpp's own CPU and Vulkan builds are 0.02 apart); a question takes 31 ms at 103 tokens, 37 ms at 256
and 109 ms at 1024 through the server (d3-lite; d3-nano 50 / 57 / 186 ms; the iGPU with llama.cpp's Vulkan build: 24 /
37 / 168 ms and 42 / 74 / 315 ms, at ~4x the energy above idle: [NPU vs
iGPU](../npu-vs-igpu.md#prefill-only-work-decisions-embeddings-rerank)).

With their mmproj (`--mmproj`; the router pairs it by itself) the d3 models read pictures in a request's state
([Decisions](../server.md#decisions)): their Qwen3-VL towers are Qwen3.5-0.8B's (12 layers of 768, 12 heads) and
Qwen3.5-2B's (24 layers of 1024), on the same encoder as the larger models' ([below](#images-qwen35--36--38)); a
picture takes 64 to 1600 tokens (the files' limits). On 57 questions about pictures (16 requests: photos of 128 to 1600
tokens, drawn shapes, an invoice; one or two pictures, pictures alone, a chat state) the answers are llama-server's
(CPU) on 56 (d3-lite) and 57 (d3-nano), the probabilities within 0.12 and 0.10 (median 0.003, 0.001): as close as
with text (the language model alone, fed llama.cpp's own image rows, differs as much on a few questions), the
encoder's bfp16 rounding adding as much on others. A question about a 512 x 512 / 1024 x 768 / 1280 x 1280 picture
takes 68 / 216 / 487 ms through the server (d3-nano 104 / 428 / 1071 ms; the iGPU 101 / 367 / 990 and 241 / 820 / 2242
ms at 3-4x the energy above idle), 8 questions about a 512 x 512 picture 342 ms (d3-nano 561; the iGPU 854, 2273).

Kev-4B (Qwen3.5-4B's shapes, causal) answers with its classifier head (`cls.output`) on the rows at the options' end
tokens and at the question's last token. Its hidden size, 2560, is two and a half K chunks of 1024: the projections that
read the residual stream take inputs of 2560 (the prompt's GEMMs read 2.5 of a row's 3 tiles: 2.77 instead of 3.16 GB
of weights a pass), and its weights go to the NPU without the LM head (3.16 GB for the 2.8 GB file). On the same
320 questions its choices are llama.cpp's on 317 and its accuracy the same within one question of 50 (median |dp|
0.0035); a question takes 83 ms at 32 tokens, 116 at 184 and 425 at 952 through the server, 8 questions about one
state 265 ms (the iGPU: 46 / 145 / 744 ms, 1379 ms), at 4-12x less energy above idle.

## Mixture of experts (Qwen3.6-35B-A3B)

Each layer's FFN is 256 experts (8 per token) and a shared expert. A decode step computes the router's logits in fp32
on every column, then each column's cores pick every token's top 8 themselves and point the column's weight stream at
the experts they take (control packets the column writes and its shim reads back); each column computes whole experts
or halves of them, and the next phase adds the columns' results into the residual stream (`moe/route.py`,
`decode/design.py`). A step streams 1.72 GB of the image at one token (8 of the 256 experts in every layer and the
screened head's part; Occamy's file, its head whole, 1.83 GB).
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

Swift 1.5's file (APEX-I-MiniPlus V3, a Qwen3.8-27B fine-tune) keeps the FFN downs in IQ4_XS and the gate / up of the
first and last sixteen layers in IQ3_S, of the middle thirty-two in IQ3_XXS, the GDN alpha in F32 and its MTP layer in
Q4_0 / Q4_1. Its 9280-byte tiles hold 16 IQ4_XS rows, 20 IQ3_S rows or 23 IQ3_XXS rows. IQ4_XS and IQ4_NL map 4-bit codes
through a table of 16 values: the cores look up two codes per byte by gathers; IQ3_S indexes a 512-entry grid of 4
values each (`formats/iq4.py`, `formats/iq3s.py`). With ten tile formats its decode design loads every format's kernel
as a code page, each IQ format's page bringing its lookup table. With its MTP drafter a step decodes up to 8 tokens
(`--spec-type none`: up to 4, a design 2% faster per step).

unsloth's dynamic quantizations of Qwen3.8-27B (UD-IQ4_XS, UD-IQ3_XXS and UD-IQ1_M tested) choose a format per tensor:
most of a layer in IQ4_XS, Q4_K, Q5_K or Q6_K, some FFN tensors in IQ3_S, IQ3_XXS, Q3_K, IQ2_S, IQ2_XS or IQ2_XXS, and
in a third of the layers the FFN gate and up in two different formats. Such a layer's gate / up tiles hold, per group of
64 outputs, its gate rows in tiles of one format, then its up rows in tiles of the other (`decode.image.split_rows`): a
decode step keeps the gate rows' outputs until the up rows' come (`decode.design`'s `gusplit`), a prompt runs a GEMM per
format and then the SiLU(gate) up step (`prefill/gate_up.py`). The IQ2 formats index grids of 8 values each, from {8,
25, 43} (256, 512 and 1024 entries): the cores look an entry up as two 8-bit masks (IQ2_XXS's as its 8 values,
`formats/iq2.py`), 24 rows per tile. The files from UD-IQ3_S down hold IQ1_S and IQ1_M tensors too (UD-IQ3_XXS a few FFN
tensors, UD-IQ1_M 42% of its bytes): their grid has 2048 entries of 8 values from {-1, 0, 1}, 32 rows per tile
(`formats/iq1.py`). An IQ1_S tile holds each block's grid entry itself (its GGUF blocks leave the room in a tile), an
IQ1_M tile the indices: the cores look those up in a table twice the size of a core's table room, so each entry is
stored once and read by two lanes. More formats than a design of 4 slots pages: these files decode up to 8 tokens per
dispatch. Their IQ tiles keep the cores busier than their bytes keep the DMA, so the smaller files are not
proportionally faster: without a drafter a step of UD-IQ4_XS (14.4 GB of weights on the NPU) takes 308 ms, of UD-IQ3_XXS
(11.5 GB) 296 ms, of UD-IQ1_M (8.7 GB) 253 ms.

## Multi-token prediction (Qwen3.8-27B, Qwen3.6-35B-A3B, Occamy 1.0)

With its drafter (the model's MTP layer: an attention layer and an FFN of its own - Qwen3.8-27B's dense, in its model
file; the 35B-A3B's an MoE layer, in a file of its own, with its own LM head), a generation step takes up to three draft
tokens along (Qwen3.8-27B and Swift 1.5 up to seven; `qwen35/draft.py`, `OverlayModel.generate`): the drafter's guesses of what
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

Qwen3.8-27B drafts with z-lab's DFlash2 block drafter where its file is in the models directory (`--spec-type draft-mtp`: with its MTP layer, whose
263 MB the image otherwise leaves out, as with `--spec-type none`; the drafter takes 1.07 GB; `artlite/decode/dflash/`). Instead of a chain of drafts, one dispatch guesses a whole
block of 7 tokens at once for ~31 ms (~11 ms per draft for the MTP layer), on a design of its own, which adds under 1 s to a load and no visible time
to a prompt. Its cache takes the model's hidden states at layers 6, 20, 34, 48 and 62 of the last 2048 positions, from each step and prompt
chunk; its head is the first 84480 rows of the model's LM head (the special tokens, the first ids and 512 of the sequence's own tokens: a few
rows' cost, for the rare tokens that end a chain), and a selector on the host picks the block's path among each position's 16 best logits.
Where the selector's first guess is uncertain (its second candidate over twice as likely as the whole chain) and the branch fits in the 8
positions that hold the round's token (its attention record and GDN block), the round's step gives the chain's last two slots to a branch:
the second candidate and its successor, after the round's token (the attention and GDN cores compute each branch slot as a step of one at
its position would, without the chain's later drafts: `QwenOverlay.step`'s branch; greedy text is the same with and without drafts). When the first draft is wrong and the branch right, the round keeps
the branch's tokens and the one after them, from the branch's own logits. On prose a quarter to a third of the first drafts are wrong:
on the same texts branches save 5% of the rounds (32K prompts, the sampler held to each prompt's text: 6.73 to 7.05 tokens/s; essays
+5%, general text +4%, JSON +4%), while code, math and thinking rarely branch. A block's cost pays on 256 greedy tokens after short
prompts: 1.4x the MTP layer's rate on code (23.5 tokens/s) and thinking, 1.6x on math, 1.3x on JSON, 1.2x on general text, 1.04x on
essays without branches. On long prose it is ahead of the MTP layer (32K-token prompts: 6.73 against 6.64 tokens/s on three prompts) and
slightly ahead on recorded agent sessions (11.8 against 11.6), whose text reaches back past its window. It takes no drafts from the context:
wherever the context could guess, the block already does. A prompt cache's checkpoints hold its cache (27 MB each). A model with images
(an mmproj file) drafts without it: its positions in an image are not the drafter's.

Every qwen35 model, with a drafter or not, also takes draft tokens from the prompt where the text repeats it
(`models/drafts.py`, [artlite-cli](../cli.md#text)); a step of them leaves the GDN states uncommitted as a drafter's
does, and with a drafter its rows follow the step's kept tokens.

## Images (Qwen3.5 / 3.6 / 3.8)

```bash
artlite-cli -m lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf \
    --mmproj lmstudio-community/Qwen3.6-35B-A3B-GGUF/mmproj-Qwen3.6-35B-A3B-BF16.gguf --image photo.jpg -p "Describe the image."
```

The Qwen models' vision encoder is their `mmproj-*.gguf` (projector `qwen3vl_merger`; BF16, F16 or Q8_0, tested:
lmstudio-community's BF16 files of Qwen3.6-35B-A3B, Qwen3.6-27B and Qwen3.8-27B, IsValorum's Q8_0 for Occamy 1.0,
ggml-org's Q8_0 for d3-lite and d3-nano: the one of the model's repository, whose width is the model's;
[artlite-server](../server.md)'s router mode pairs it with the model by itself). An image is resized whole, as
transformers' processor and llama.cpp resize it: its sides to multiples of 32 pixels at its aspect, between
`--image-min-tokens` and `--image-max-tokens` tokens of 32 x 32 pixels (default the mmproj's limits where it has them,
d3's 64 and 1600, else llama.cpp's 8 and 4096; a 1024 x 768 photo is 768 tokens, a phone's 12-megapixel photo 4015). Its tokens take the
official M-RoPE positions (rows and columns from the position after `<|vision_start|>`; llama.cpp's are the same).

The encoder (`qwen3vl/encoder.py`) runs on the vision overlay (in a hardware context of its own): 27 layers of
attention over all of the image's 16 x 16 patches (a 2-D RoPE in each head; Qwen3.5-0.8B's and 2B's smaller towers, d3's:
12 layers of 768 and 24 of 1024), then each 2 x 2 window merged into one token by an MLP. Its time grows with the
patches and, through the attention, with their square (d3-lite's tower 0.023 s at 512 x 512, d3-nano's 0.057 s):

| image (tokens) | 512 x 512 (256) | 1024 x 1024 (1024) | 2048 x 2048 (4096) |
|---|---|---|---|
| Artlite (NPU) | 0.066 s | 0.39 s | 4.0 s |
| llama.cpp, iGPU (Vulkan) | 0.27 s | 1.33 s | 10.3 s |
| llama.cpp, CPU (16 threads) | 0.42 s | 2.61 s | 27.1 s |

(Strix Halo, Qwen3.6-35B-A3B's mmproj; llama.cpp's `mtmd batch encoding` time.)

A large image runs as a dispatch per one or a few layers (each well under the driver's 2 s). Its weights are converted
to bfp16 at every load (the patch embedding's and the merger's MLP's as bfp16 hi + lo pairs; the pixels enter as
their exact 8-bit values), the dispatches of each image size cached (~15 MB, the cache directory; later loads ~0.25 s); an image's
embeddings are kept, so a chat's next turns do not run it again. The embeddings are ~4% per token from an exact
(float64) encoder, llama.cpp's ~0.7%; a few tokens are much further off in both, wherever the encoder's last layer
half switches on a huge activation, and the language model hardly notices: its answer stays close to llama.cpp's own
(teacher-forced over a description: KL 0.006, argmax agreement 0.95).

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
per entry. Its weights are converted to bfp16 at every load (the most sensitive ones, the patch embedding's and the
final merger's, as bfp16 hi + lo pairs; the pixels enter as their exact 8-bit values), the dispatches cached as Stable
Diffusion's (~16 MB, the cache directory; later loads ~0.3 s); an image's embeddings are kept, so a chat's next
turns do not run it again.

