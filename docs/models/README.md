# Models

How each model runs on the NPU, its files and its measurements: [Llama 3](llama3.md), [Gemma 4](gemma4.md),
[Qwen3.5 / 3.6 / 3.8, MiniCPM-V 4.7](qwen35.md), [Stable Diffusion 1.5](sd15.md).

Model files are found by path, or relative to the models directory (`ARTLITE_MODELS_DIR`, default
`~/.artlite/models`) as `<account>/<repository>/<file>`: [Installation](../install.md#models-directory). A GGUF file's
architecture, tokenizer and chat template come from its metadata.

## Tested files and plans

✅ runs on the NPU, 🤔 planned. Files are named by Hugging Face account and quant.

| architecture | model | files | planned |
|---|---|---|---|
| llama | ✅ Llama 3.2 1B Instruct | unsloth Q4_0, Q4_1, Q4_K_M, BF16; lmstudio-community Q3_K_L; bartowski Q6_K; MaziyarPanahi Q5_K_M | lmstudio-community's `model.safetensors` |
| | ✅ Llama 3.2 3B Instruct | unsloth Q4_K_M, BF16; bartowski Q4_0, Q5_K_M; MaziyarPanahi Q2_K, Q3_K_M | |
| | ✅ Llama 3.1 8B Instruct | lmstudio-community Q4_K_M; bartowski Q4_K_S | EAGLE3 draft head (`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`) |
| | 🤔 Llama 3.3 70B's EAGLE3 draft head | | `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B` `pytorch_model.bin` |
| gemma4 | ✅ Gemma 4 E4B it | ggml-org Q4_0, with its MTP drafter | vision + audio; ggml-org Q8_0, BF16 (with MTP); lmstudio-community Q4_K_M; unsloth Q4_1; HauhauCS Uncensored Aggressive Q4_K_M, Q2_K_P |
| | 🤔 Gemma 4 12B it | | lmstudio-community Q4_K_M, QAT Q4_0; unsloth Q4_K_M (with MTP); vision + audio |
| | 🤔 Gemma 4 26B-A4B it | | lmstudio-community QAT Q4_0; vision |
| | 🤔 Gemma 4 31B it | | lmstudio-community QAT Q4_0; HauhauCS QAT Uncensored Balanced Q4_K_M (with MTP); vision |
| qwen35 | ✅ Qwen3.5 9B | lmstudio-community Q4_K_M | |
| | ✅ Qwen3.6 27B | lmstudio-community Q4_K_M | vision |
| | ✅ Qwen3.8 27B | lmstudio-community Q4_K_M | MTP, vision |
| | ✅ MiniCPM-V 4.7 1B | openbmb Q4_K_M, its mmproj f16 (16x) | the 4x mmproj; openbmb's other quants |
| | 🤔 Bonsai 27B (1-bit) | | lmstudio-community Q1_0; vision |
| | 🤔 Ternary Bonsai 2 27B | | prism-ml PTQ1_0, PQ2_0; vision |
| dspark | 🤔 Ternary Bonsai 27B's drafter | | prism-ml Q4_1; vision |
| dflash | 🤔 MiniCPM-V 4.7 1B's DSpark drafter | | openbmb F16, Q4_K_M |
| qwen3 | 🤔 Qwen3 0.6B | | unsloth Q4_1 |
| text to image | ✅ Stable Diffusion 1.5 | stable-diffusion-v1-5 diffusers F32 (safetensors) | the text encoder on the NPU |
| speech recognition | 🤔 Whisper base.en | | whisper.cpp `ggml-base.en.bin` |
