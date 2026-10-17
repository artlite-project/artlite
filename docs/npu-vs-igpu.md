# Where the NPU fits

In a Ryzen AI processor the CPU, the iGPU and the NPU share one memory bus and one power budget. Which work should
run on the NPU rather than the iGPU is an open question; this page gives what is measured and what is left to try.
The measurements are on a Ryzen AI Max+ 395 (Strix Halo: 256-bit LPDDR5X-8000, 256 GB/s; Radeon 8060S iGPU), NPU
power mode `default`.

## Memory bandwidth

All of the NPU's DRAM traffic goes through one 32-byte-per-cycle port to the SoC's Infinity Fabric, clocked by the AI
Engine array: ~57 GB/s at 1.81 GHz, for reads and for writes alike. One shim DMA channel streams 13.8 GB/s, one shim
column 27 GB/s, and four channels fill the port: no kernel, data layout or DMA setting gets past it. Its rate follows
the array clock, so the NPU's power modes scale it (`balanced`: ~38 GB/s, `powersaver`: ~23 GB/s).

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
  per weight pass (a step of 4 tokens costs ~5% more than a step of one: multi-token prediction, speculative
  decoding) and fewer bytes per token (MoE, lower-bit weights).
- **Prefill** is compute-bound and on par, at a fraction of the iGPU's power.
- **Both decoding at once** reach ~204 GB/s in total, 7% above the iGPU alone: they compete for the bus.
- **Smaller platforms.** Strix Point and Krackan Point have a 128-bit bus (~120 GB/s peak) while the NPU keeps its own
  limit: the iGPU's decode lead should shrink to about 1.8x (not measured). Under a laptop's power cap or on
  battery, the iGPU cannot hold its clocks.

## Candidates to explore

So the NPU can win where compute rather than DRAM bandwidth bounds the work, or where watts or a free GPU matter more
than speed. Not tried yet, ranked by expected value per effort:

| | candidate | why the NPU could win |
|---|---|---|
| 1 | Prefill-only work: an LLM as a classifier, decision models (JEV-27B-VL), rerankers, guard models; prefill-heavy agents and RAG | compute-bound; prefill on par at a fraction of the power |
| 2 | Prefill on the NPU, decode on the iGPU | each device on its strength, overlapped in a server |
| 3 | Multimodal encoders (vision and audio projectors, Whisper) | ViT and conformer encoders are compute-bound |
| 4 | Image generation (UNet, DiT) | many compute-bound steps |
| 5 | Diffusion LLMs (parallel block decoding) | a block of tokens per weight pass turns decode into prefill |
| 6 | Batched decode, parallel sampling | the NPU's spare compute in decode |
| 7 | Extremely compressed models (1-bit, ternary) | few bytes per token; the iGPU may become bound by unpacking them |
| 8 | Power-capped and smaller platforms | the iGPU throttles under a cap; a 128-bit bus narrows its lead |
| 9 | An always-on second model beside the iGPU's | the GPU stays free; the two share only DRAM |
