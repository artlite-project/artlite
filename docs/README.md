# Artlite documentation

| | |
|---|---|
| [Installation](install.md) | hardware, driver, build, models directory, troubleshooting |
| [artlite-cli](cli.md) | text and images from the command line |
| [artlite-server](server.md) | the OpenAI-compatible server and web chat |
| [artlite-bench](bench.md) | prompt and generation speeds, as llama-bench |
| [Models](models/README.md) | tested files, plans, and how each model runs on the NPU: [Llama 3](models/llama3.md), [Gemma 4](models/gemma4.md), [Qwen3.5 / 3.6 / 3.8](models/qwen35.md), [Stable Diffusion 1.5](models/sd15.md) |
| [Benchmark methodology](methodology.md) | how the README's charts are measured: test system, files, other runtimes |
| [Where the NPU fits](npu-vs-igpu.md) | bandwidth and power against the iGPU, workloads to explore |
| [Design](design.md) | runtime layers, C++ and Python API, kernel compilation, error handling, driver behaviour |
| [Development](development.md) | tests, operators, kernel tools, environment variables, skills |

`data/` and `charts.py` hold the performance charts' numbers and draw them into `images/`; `diagram.py` draws the
README's how-it-works diagram.
