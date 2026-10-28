# Where the NPU fits

In a Ryzen AI processor the CPU, the iGPU and the NPU share one memory bus and one power budget. Which work should
run on the NPU rather than the iGPU is an open question; this page gives what is measured and what is left to try.
The measurements are on a Ryzen AI Max+ 395 (Strix Halo: 256-bit LPDDR5X-8000, 256 GB/s; Radeon 8060S iGPU), NPU
power mode `default`.

## Memory bandwidth

On Strix Halo all of the NPU's DRAM traffic goes through one 32-byte-per-cycle port to the SoC's Infinity Fabric,
clocked by the AI Engine array: ~57 GB/s at 1.81 GHz, for reads and for writes alike. One shim DMA channel streams
13.8 GB/s, one shim column 27 GB/s, and four channels fill the port: no kernel, data layout or DMA setting gets past
it. Its rate follows the array clock, so the NPU's power modes scale it (`balanced`: ~38 GB/s, `powersaver`: ~23
GB/s). The port is Strix Halo's, not the array's. A Krackan Point laptop's NPU (Ryzen AI 7 350, 128-bit LPDDR5X-7500,
~120 GB/s) reads 97.8 GB/s and writes 87.2: 81% of its bus and 1.7x Strix Halo's NPU. It is the same 8-column array at
the same 1.81 GHz, with the same rates per DMA channel and per shim. Its decode reads 64-70 GB/s so far (Qwen3.8 27B:
4.2 tokens/s), ~70% of that, where Strix Halo's decode reaches 95-97% of its port. Two causes are likely: each
column streams its weights through one of its shim's two channels, and for K-quant weights the cores' work per tile
comes next. `artlite-probe` measures a machine's clock and rates ([both machines' numbers](probe.md)).

| DRAM reads, GB/s | alone | NPU + iGPU together |
|---|---|---|
| NPU (all 16 shim channels) | 56.5 | 40.9 |
| iGPU (Vulkan compute) | 243.6 | 198.4 |
| CPU (16 cores; each CCD's link: 64 GB/s) | 120.1 | |

Any mix with the iGPU stops at 236-240 GB/s, the bus's practical ceiling, and the iGPU wins the arbitration (the CPU
keeps a quarter of its solo rate beside it).

## Decode and prefill

Qwen3.5 9B and Qwen3.6 27B Q4_K_M; Artlite's decode at one token per step, against llama.cpp's Vulkan backend
(`llama-bench -ngl 99`) on the same files:

| | NPU (Artlite) | iGPU (llama.cpp Vulkan) |
|---|---|---|
| 9B decode | 10.4 tokens/s | 37.7 tokens/s |
| 27B decode | 3.37 tokens/s | 12.6 tokens/s |
| read rate in decode | 52.5-53.3 GB/s | 190-199 GB/s |
| 9B prefill, 2048 tokens | ~960 tokens/s | 987 tokens/s |
| power over idle, 9B decode | ~8 W (~0.8 J per token) | ~38 W (~1.0 J per token) |

- **Decode** reads every weight once per token, so each device decodes at its own read limit and the tokens/s ratio
  (3.6-3.7x) is the bandwidth ratio. Artlite's decode is close to the NPU's limit already: what remains is more tokens
  per weight pass (a step of 4 tokens costs ~5% more than a step of one: the draft tokens Artlite's generation takes
  from a drafter or from the context) and fewer bytes per token (MoE, lower-bit weights).
- **Prefill** is compute-bound and on par with llama.cpp, at a fraction of the iGPU's power. Engines tuned for this
  iGPU go further: [a gufo fork](https://github.com/mikealanni/gufo) publishes GPU-only rates at 8192-token prompts of
  1.4x Artlite's on Qwen3.8 27B (629 vs 436 tokens/s) and 1.5x on Qwen3.6 35B-A3B (2862 vs 1888; UD-Q4_K_XL files).
- **Agents** are decode-bound once a prompt cache keeps their conversations: in recorded sessions of Claude Code,
  Codex and opencode on Qwen3.6 35B-A3B, the turns read 5.5 new prompt tokens per generated token, and 70% of the
  generated tokens were thoughts. From both devices' measured rates (llama.cpp's Vulkan backend on the iGPU), the iGPU
  would finish those sessions ~1.85x sooner on Strix Halo (the NPU prefills 2.3x faster, decodes 2.6x slower); at a
  Krackan Point's rates (Gemma 4 E4B's, an estimate) the two finish about together, the NPU ~1.3x sooner without the
  thoughts. On Strix Halo the NPU's case for agents is the free iGPU and the energy.
- **Both decoding at once** reach ~204 GB/s in total, 7% above the iGPU alone: they compete for the bus.
- **Smaller platforms.** Strix Point and Krackan Point have a 128-bit bus (~90 GB/s with DDR5-5600, ~120 with
  LPDDR5X-7500). On a Krackan Point laptop
  (Ryzen AI 7 350, LPDDR5X-7500, Radeon 860M) llama.cpp's Vulkan decode leads Artlite's by 1.44x (Llama 3.2 1B
  Q4_K_M: 113.5 vs 78.7 tokens/s), 1.25x (Gemma 4 E4B Q4_0: 29.3 vs 23.4) and 1.36x (Qwen3.8 27B Q4_K_M: 5.77 vs
  4.24), plugged in and on battery alike: that laptop's iGPU holds its clocks on battery.

## Candidates to explore

So the NPU can win where compute rather than DRAM bandwidth bounds the work, or where watts or a free GPU matter more
than speed. Not tried yet, ranked by expected value per effort:

| | candidate | why the NPU could win |
|---|---|---|
| 1 | Prefill-only work: an LLM as a classifier, decision models (JEV-27B-VL), rerankers, guard models; RAG | compute-bound; prefill at a fraction of the power |
| 2 | Prefill on the NPU, decode on the iGPU | each device on its strength, overlapped in a server |
| 3 | Multimodal encoders (vision and audio projectors, Whisper) | ViT and conformer encoders are compute-bound |
| 4 | Image generation (UNet, DiT) | many compute-bound steps |
| 5 | Diffusion LLMs (parallel block decoding) | a block of tokens per weight pass turns decode into prefill |
| 6 | Batched decode, parallel sampling | the NPU's spare compute in decode |
| 7 | Extremely compressed models (1-bit, ternary) | few bytes per token; the iGPU may become bound by unpacking them |
| 8 | Power-capped and smaller platforms | a 128-bit bus narrows the iGPU's decode lead to 1.25-1.44x (Krackan Point) |
| 9 | An always-on second model beside the iGPU's | the GPU stays free; the two share only DRAM |
