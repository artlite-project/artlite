# Models

How each model runs on the NPU, its files and its measurements: [Llama 3](llama3.md), [Qwen3](qwen3.md),
[Gemma 4](gemma4.md), [Qwen3.5 / 3.6 / 3.8 (and Qwen3.6 35B-A3B), MiniCPM-V 4.7](qwen35.md), [LFM3 (and LFM3-VL-2B, LFM3-Audio-2B, LFM3-Omni-2B, d1-2B, d1-Omni-2B)](lfm3.md),
[Spark-X2.6 4B, 1.7B](spark.md), [Flint](flint.md), [Inkling 2 Mini](inkling2.md),
[MOSS-Transcribe-Diarize (speech)](moss.md), [Stable Diffusion 1.5](sd15.md).

Model files are found by path, or relative to the models directory (`ARTLITE_MODELS_DIR`, default
`~/.artlite/models`) as `<account>/<repository>/<file>`: [Installation](../install.md#models-directory). A GGUF file's
architecture, tokenizer and chat template come from its metadata.

A Hugging Face checkpoint directory (`config.json`, safetensors, `tokenizer.json`) is a model file too: `-m
<account>/<repository>` opens it as llama.cpp would open the GGUF its converter makes of it (the same metadata,
vocabulary, chat template and weights, read from the safetensors at every load: no converted copy). Supported:
`Qwen3ForCausalLM`, `LlamaForCausalLM`, `Qwen3_5ForConditionalGeneration` (its MTP block as the drafter),
`MossTranscribeDiarizeForConditionalGeneration` (speech: [MOSS-Transcribe-Diarize](moss.md)); BF16, F16 or F32
weights run as BFP16 tiles, as a BF16 GGUF does. Tested: Qwen/Qwen3-0.6B, Qwen/Qwen3.5-0.8B,
lmstudio-community/Llama-3.2-1B-Instruct (the same logits as their converted GGUFs), OpenMOSS-Team/MOSS-Transcribe-Diarize
(a clip's transcript).

ParoQuant (PARO) checkpoints, int4 or int5 (`int5-bitplane`) with groups of 128 (`quantization_config.quant_method`
`paroquant`), run as checkpoint directories of `Qwen3ForCausalLM`, `LlamaForCausalLM`,
`Qwen3_5ForConditionalGeneration` (its text; int5: Qwen3.5's), `Qwen3_5MoeForConditionalGeneration` (its experts PARO,
its router and shared expert F16) and `Gemma4ForConditionalGeneration` (its text; E4B): their linears as 4-bit (5-bit) tiles of the checkpoint's size (4.5, 5.5 bits per
weight with their scales), each projection's input rotated on the NPU as the checkpoint prescribes (its channel scales
and Givens rounds), the head and embeddings from the checkpoint's F16 tensors; a Qwen3.5 / 3.6 checkpoint's vision
tower with `--mmproj` the directory ([images](qwen35.md#images-qwen35--36--38)). Tested: z-lab/Qwen3-0.6B-PARO (KL 0.0011
against the checkpoint computed exactly on the CPU), z-lab/Llama-3.1-8B-Instruct-PARO (its first layers as the
checkpoint computes them; a step of one token 81.7 ms, bartowski's Q4_K_S GGUF 76.0), z-lab/Qwen3.5-0.8B-PARO (KL 0.0003), Launch80/Qwen3.8-27B-PARO-int5
(its first layers within 0.6% of the checkpoint computed exactly; with its base model's DFlash2 drafter,
z-lab/Qwen3.8-27B-DFlash2-GGUF, found and used as for the GGUF files, on steps of up to 8 tokens, the same greedy text
with and without drafts), z-lab/Qwen3.8-27B-PARO (int4: its first layers within 0.5%; the same drafter),
z-lab/Qwen3.6-35B-A3B-PARO (its first layers within 0.6%; with its base model's MTP drafter,
ggml-org's mtp-Qwen3.6-35B-A3B-Q4_0.gguf, found and used as for the GGUF files, the same greedy text),
z-lab/gemma-4-E4B-it-PARO (teacher-forced KL 0.00029 against the checkpoint computed exactly; a step of one token 53.9
ms, ggml-org's Q4_0 GGUF 55.6, at 10% fewer bytes a token). Their speeds next to the GGUF files' (the
[charts](../../README.md#performance)' runs, [methodology](../methodology.md)):

| | TTFT 1K / 32K | decode 1K / 32K | with the drafter 1K / 32K (S: tokens a step) |
|---|---|---|---|
| Qwen3.8 27B, PARO int5 | 3.27 / 120 s | 3.0 / 2.8 tokens/s | 8.9 / 6.4 tokens/s (S 8) |
| Qwen3.8 27B, Q4_K_M | 2.25 / 86.4 s | 3.3 / 3.0 | 8.6 / 6.5 (S 8) |
| Qwen3.6 35B-A3B, PARO int4 | 0.825 / 21.4 s | 25.6 / 21.5 | 35.0 / 29.9 (S 4 / 2) |
| Qwen3.6 35B-A3B, Q4_K_M | 0.811 / 21.8 s | 27.4 / 23.5 | 35.8 / 30.5 (S 3) |
| Gemma 4 E4B, PARO int4 | 0.341 / 12.4 s | 17.8 / 14.3 | |
| Gemma 4 E4B, Q4_0 | 0.38 / 12.9 s | 16.7 / 13.7 | 33.0 / 20.4 (S 4 / 2) |

The 27B int5's prompts are slower than its GGUF's: its 5-bit tiles have Q5_K's planar layout, which the prefill
multiplies on three of a column's four cores (the fourth dequantizes), where the GGUF's Q4_K tiles use all four, and a
chunk is compute bound.

## Tested files and plans

✅ runs on the NPU, 🤔 planned. Files are named by Hugging Face account and quant.

| architecture | model | files | planned |
|---|---|---|---|
| llama | ✅ Llama 3.2 1B Instruct | unsloth Q4_0, Q4_1, Q4_K_M, BF16; lmstudio-community Q3_K_L, its checkpoint directory (BF16); bartowski Q6_K; MaziyarPanahi Q5_K_M | |
| | ✅ Llama 3.2 3B Instruct | unsloth Q4_K_M, BF16; bartowski Q4_0, Q5_K_M; MaziyarPanahi Q2_K, Q3_K_M | |
| | ✅ Llama 3.1 8B Instruct | lmstudio-community Q4_K_M; bartowski Q4_K_S; z-lab's PARO checkpoint (int4) | EAGLE3 draft head (`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`) |
| | 🤔 Llama 3.3 70B's EAGLE3 draft head | | `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B` `pytorch_model.bin` |
| gemma4 | ✅ Gemma 4 E4B it | ggml-org Q4_0, with its MTP drafter; z-lab's PARO checkpoint (int4) | vision + audio; ggml-org Q8_0, BF16 (with MTP); lmstudio-community Q4_K_M; unsloth Q4_1; HauhauCS Uncensored Aggressive Q4_K_M, Q2_K_P |
| | 🤔 Gemma 4 12B it | | lmstudio-community Q4_K_M, QAT Q4_0; unsloth Q4_K_M (with MTP); vision + audio |
| | 🤔 Gemma 4 26B-A4B it | | lmstudio-community QAT Q4_0; vision |
| | 🤔 Gemma 4 31B it | | lmstudio-community QAT Q4_0; HauhauCS QAT Uncensored Balanced Q4_K_M (with MTP); vision |
| qwen35 | ✅ Qwen3.5 9B | lmstudio-community Q4_K_M | |
| | ✅ Qwen3.5 0.8B | Qwen's checkpoint directory (BF16, with its MTP block); z-lab's PARO checkpoint (int4) | |
| | ✅ Qwen3.6 27B | lmstudio-community Q4_K_M, its mmproj BF16 (images) | |
| | ✅ Qwen3.8 27B | lmstudio-community Q4_K_M, with its MTP layer as the drafter; its mmproj BF16 (images); unsloth UD-IQ4_XS (gate / up in two formats, IQ2 tensors), UD-IQ3_XXS, UD-IQ1_M (IQ1_S, IQ1_M tensors); Launch80's PARO checkpoint (int5) | |
| | ✅ Swift 1.5 (Qwen3.8 27B fine-tune, uncensored) | IsValorum APEX-I-MiniPlus V3 (IQ4_XS / IQ3_S / IQ3_XXS FFN), with its MTP layer as the drafter | its mmproj Q8_0 |
| | ✅ MiniCPM-V 4.7 1B | openbmb Q4_K_M, its mmproj f16 (16x, 4x) | openbmb's other quants |
| | ✅ vllm-sr d3-lite, d3-nano (decision models: [`/v1/systemone`](../server.md#decisions)) | ggml-org Q8_0, their mmproj Q8_0 (images) | |
| | ✅ Kev-4B (decision model: [`/v1/systemone`](../server.md#decisions)) | ggml-org Q4_K_M | |
| qwen35moe | ✅ Qwen3.6 35B-A3B | lmstudio-community Q4_K_M, with ggml-org's MTP drafter (Q4_0); its mmproj BF16 (images); z-lab's PARO checkpoint (int4) | |
| | ✅ Occamy 1.0 (Qwen3.6 35B-A3B fine-tune, abliterated) | IsValorum APEX-I-MiniPlus V2.1 (IQ3_XXS / Q3_K experts), with its MTP drafter (Q4_K_M); its mmproj Q8_0 (images) | |
| | 🤔 Bonsai 27B (1-bit) | | lmstudio-community Q1_0; vision |
| | 🤔 Ternary Bonsai 2 27B | | prism-ml PTQ1_0, PQ2_0; vision |
| dspark | 🤔 Ternary Bonsai 27B's drafter | | prism-ml Q4_1; vision |
| dflash | ✅ Qwen3.8 27B's DFlash2 drafter | z-lab Q4_K_M | |
| | ✅ Flint's DFlash2 drafter | incoai Q4_K_M | |
| | ✅ Spark-X2.6 4B's and 1.7B's DSpark drafters | XHToken BF16 | |
| | 🤔 MiniCPM-V 4.7 1B's DSpark drafter | | openbmb F16, Q4_K_M |
| qwen3 | ✅ Qwen3 0.6B | unsloth Q4_1; Qwen's checkpoint directory (BF16); z-lab's PARO checkpoint (int4) | |
| | ✅ Qwen3-Embedding-0.6B ([embeddings](../server.md#embeddings-and-rerank)) | Qwen Q8_0 | |
| | ✅ Qwen3-Embedding-4B ([embeddings](../server.md#embeddings-and-rerank)) | Qwen Q4_K_M | |
| | ✅ Qwen3-Reranker-0.6B ([rerank](../server.md#embeddings-and-rerank)) | ggml-org Q8_0 | |
| | ✅ Qwen3-Reranker-4B ([rerank](../server.md#embeddings-and-rerank)) | giladgd Q4_K_M | |
| vllm-decision2 | ✅ vllm-sr Decision-2.0-Kai-0.6B (decision model: [`/v1/systemone`](../server.md#decisions)) | Q8_0 by llama.cpp's converter from vllm-sr's checkpoint | |
| lfm3 | ✅ LFM3-1.7B | LiquidAI QAD-Q4_0, Q4_K_M, Q8_0 | the sliding-window layers' caches as rings; LiquidAI BF16 |
| | ✅ d1-2B (decision model: [`/v1/systemone`](../server.md#decisions)) | LiquidAI Q4_0, Q4_K_M, Q8_0, its mmproj Q8_0 (images, videos) | |
| | ✅ LFM3-VL-2B | LiquidAI QAD-Q4_0, its mmproj BF16, Q8_0 ([images](lfm3.md#lfm3-vl-2b-images), [videos](lfm3.md#videos)) | LiquidAI Q4_K_M, Q8_0 |
| | ✅ LFM3-Audio-2B (speech in: [transcription, live transcription](lfm3.md#lfm3-audio-2b-speech); speech out: [text to speech, spoken replies](lfm3.md#speech-output)) | LiquidAI Q4_K_M, Q4_0, its mmproj Q8_0 | LiquidAI Q8_0 |
| | ✅ LFM3-Omni-2B ([images, videos with their sound, speech in and out](lfm3.md#lfm3-omni-2b-images-videos-speech)) | LiquidAI Q4_K_M, Q4_0, its mmproj Q8_0 | LiquidAI Q8_0 |
| | ✅ d1-Omni-2B (decision model: [`/v1/systemone`](../server.md#decisions); audio clips, spoken questions) | LiquidAI Q8_0, its mmproj Q8_0 | |
| moss | ✅ MOSS-Transcribe-Diarize 0.9B (speech: [transcription](moss.md)) | OpenMOSS-Team's checkpoint directory (BF16) | |
| spark2_5 | ✅ Spark-X2.6 4B | XHToken Q4_K_M, Q8_0, with its DSpark drafter; bartowski Q4_K_M, Q4_0 | Spark-X2.5 4B |
| | ✅ Spark-X2.6 1.7B | XHToken Q4_K_M, Q8_0, with its DSpark drafter | |
| flint | ✅ Flint (40B MoE, ~2.5B active) | RecursiveSI int4 (Q4_1 pairs, packed as Q4_1_64), with incoai's DFlash2 drafter (Q4_K_M) | unsloth UD-Q4_K_XL |
| inkling2_loop, inkling2 | ✅ Inkling 2 Mini (26B MoE, 8 of 128 experts, a looped core) | unsloth UD-Q4_K_XL (NVFP4 / Q8_0 experts), with its DSpark drafter, its mmproj F16 ([images](inkling2.md#images)) | |
| text to image | ✅ Stable Diffusion 1.5 | stable-diffusion-v1-5 diffusers F32 (safetensors) | the text encoder on the NPU |
