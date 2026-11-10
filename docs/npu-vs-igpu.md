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
  per weight pass (a step of 4 tokens costs ~3-8% more than a step of one, Qwen3.8 27B's step of 8 tokens 4-7%: the draft
  tokens Artlite's generation takes from a drafter or from the context; Qwen3.8 27B with its DFlash2 drafter: 8.6 tokens/s
  after a 1K-token essay prompt and 23.5 on code instead of 3.3) and fewer bytes per token (MoE, lower-bit weights).
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

## A small hybrid model's prompts: LFM3-1.7B

LFM3-1.7B (QAD-Q4_0: 13 short-convolution layers, 11 attention layers, 1.07 GB of weights) is the case where the NPU is
the faster device for every prompt length: its prefill is compute-bound GEMMs with a small KV cache (6 full-attention
layers). A chat request of an essay prompt and one generated token through each server (`artlite-server` on the NPU,
`llama-server -ngl 99 -fa on` on the iGPU), the same file, each request's text new and uncached, energy above a 10 W
idle (socket power from the SMU):

| prompt tokens | NPU | iGPU | energy a request above idle, NPU / iGPU |
|---|---|---|---|
| 1027 | 216 ms | 250 ms | 3.9 / 13.2 J |
| 8192 | 1.26 s | 2.13 s | 27 / 93 J |
| 32771 | 6.35 s | 12.07 s | 142 / 462 J |

The NPU draws 18-22 W over idle against the iGPU's 39-53 W, and uses ~3.3x less energy per prompt. Generation goes the
other way: 50.0 / 41.5 tokens/s on the NPU after 1K / 32K tokens against the iGPU's 168.8 / 122.5 (`llama-bench`).
Prompt-heavy work on this model (agents' long system prompts and re-read files, long documents) is where the NPU fits.

## Prefill-only work: decisions, embeddings, rerank

A decision, an embedding or a rerank score is one forward pass over a prompt and no generated text, so the work is
prefill. Through each server ([artlite-server's](server.md#embeddings-and-rerank) on the NPU, llama-server's Vulkan
build with `-ngl 99` on the iGPU), on the same files, one device at a time, warm, HTTP included, each request's text
new; energy per request above an 18 W idle (socket power from the SMU, idle varying by a few W with the desktop's
other programs):

| file | request | prompt tokens | NPU | iGPU | energy a request, NPU / iGPU (the shortest, the longest prompt) |
|---|---|---|---|---|---|
| vllm-sr d3-lite (0.8B, Q8_0) | a question | 103 / 256 / 1024 | 32 / 39 / 113 ms | 24 / 37 / 168 ms | 0.26 / 1.0 J, 1.4 / 6.2 J |
| | 8 questions about one state | 2315 | 282 ms | 338 ms | 0.56 / 1.4 J a question |
| vllm-sr d3-nano (2B, Q8_0) | a question | 103 / 256 / 1024 | 50 / 57 / 186 ms | 42 / 74 / 315 ms | 0.8 / 1.7 J, 3.2 / 11 J |
| Kev-4B (Q4_K_M) | a question | 32 / 184 / 952 | 90 / 125 / 453 ms | 46 / 145 / 744 ms | 0.4 / 2.0 J, 6.0 / 25 J |
| | 8 questions about one state | 1732 | 278 ms | 1379 ms | 0.48 / 5.8 J a question |
| Decision-2.0-Kai (0.6B, Q8_0) | a question | 79 / 268 / 1036 | 31 / 56 / 159 ms | 17 / 34 / 129 ms | 0.22 / 0.66 J, 2.0 / 4.6 J |
| LiquidAI d1-2B (Q4_0) | a question | 64 / 256 / 1024 | 33 / 49 / 149 ms | 22 / 69 / 298 ms | 0.36 / 0.73 J, 3.3 / 10 J |
| | 8 questions about one state | 482 (one sequence; the iGPU: 1833, each alone) | 84 ms | 490 ms | 0.23 / 1.8 J a question |
| Qwen3-Embedding-0.6B (Q8_0) | a query / a passage | 30 / 433 | 27 / 55 ms | 12 / 51 ms | 0.16 / 0.50 J, 0.72 / 1.9 J |
| | 8 passages a request | 3434 | 11.1K tokens/s | 8.0K tokens/s | 0.72 / 1.9 J a passage |
| Qwen3-Embedding-4B (Q4_K_M) | a query / a passage | 30 / 433 | 79 / 216 ms | 50 / 326 ms | 0.9 / 2.0 J, 4.4 / 11 J |
| | 8 passages a request | 3434 | 2.1K tokens/s | 1.1K tokens/s | 4.0 / 13 J a passage |
| Qwen3-Reranker-0.6B (Q8_0) | 20 pairs of ~510 tokens | 10194 | 26.1 pairs/s | 17.1 pairs/s | 0.60 / 2.5 J a pair |

- **Energy**: the NPU answers with 2-5x less energy above idle (8-12x for several questions about one state), drawing
  5-18 W over idle against the iGPU's 32-45 W, and leaves the GPU free.
- **Latency**: the iGPU answers a single short request sooner (up to ~100 tokens: its fixed cost per request is
  lower); from ~256 tokens on the two are even, and longer prompts, batches and several questions about one state
  (packed into shared prefill chunks on the NPU) go faster on the NPU, up to 6x.
- A decision model with full attention over the whole prompt (vllm-d3, d1) runs a prompt in one chunk of the next size:
  one just past a size pays the next one (d3-lite, 260 tokens: 62 ms, 39 at 256). d1 reads a state's questions in one
  sequence (its runner's way), the state once: 4.4x faster than each question alone on the NPU.

Decisions about pictures (each side with the model's mmproj, its vision encoder on the same device; every request's
picture new; energy a question above a 12.5 W idle):

| file | request | NPU | iGPU | energy a question, NPU / iGPU (the smallest, the largest picture) |
|---|---|---|---|---|
| LiquidAI d1-2B (Q4_0) | a question about a 512 x 512 / 1024 x 768 / 1280 x 1280 picture | 87 / 281 / 613 ms | 118 / 370 / 974 ms | 1.8 / 4.8 J, 8.9 / 32 J |
| | 8 questions about a 512 x 512 picture | 127 ms (one sequence) | 1241 ms | 0.31 / 5.6 J |
| vllm-sr d3-lite (Q8_0) | a question about a 512 x 512 / 1024 x 768 / 1280 x 1280 picture | 67 / 222 / 520 ms | 98 / 359 / 983 ms | 1.2 / 3.1 J, 6.1 / 30 J |
| | 8 questions about a 512 x 512 picture | 342 ms | 892 ms | 0.65 / 3.2 J |
| vllm-sr d3-nano (Q8_0) | a question about a 512 x 512 / 1024 x 768 / 1280 x 1280 picture | 104 / 449 / 1162 ms | 229 / 796 / 2099 ms | 2.1 / 7.0 J, 16 / 59 J |
| | 8 questions about a 512 x 512 picture | 563 ms | 2104 ms | 1.3 / 7.5 J |

With a picture the NPU answers 1.3-2.2x sooner than the iGPU at 2.6-4.9x less energy above idle; several questions
about one picture 2.6-10x sooner at 5-18x less.

## Candidates to explore

So the NPU can win where compute rather than DRAM bandwidth bounds the work, or where watts or a free GPU matter more
than speed. Not tried yet, ranked by expected value per effort:

| | candidate | why the NPU could win |
|---|---|---|
| 1 | Prefill on the NPU, decode on the iGPU | each device on its strength, overlapped in a server |
| 2 | Multimodal encoders (vision and audio projectors, Whisper) | ViT and conformer encoders are compute-bound |
| 3 | Image generation (UNet, DiT) | many compute-bound steps |
| 4 | Diffusion LLMs (parallel block decoding) | a block of tokens per weight pass turns decode into prefill |
| 5 | Batched decode, parallel sampling | the NPU's spare compute in decode |
| 6 | Extremely compressed models (1-bit, ternary) | few bytes per token; the iGPU may become bound by unpacking them |
| 7 | Power-capped and smaller platforms | a 128-bit bus narrows the iGPU's decode lead to 1.25-1.44x (Krackan Point) |
| 8 | An always-on second model beside the iGPU's | the GPU stays free; the two share only DRAM |
