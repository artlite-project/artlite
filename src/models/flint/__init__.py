"""Recursive's Flint on the NPU (GGUF architecture ``flint``): ``flint`` (the model); on the decode overlay
(``artlite.decode``) the weight image (``image``), the decode design (``decode``) and its host side (``overlay``), the
prompt's chunk programs (``prefill``). Its MoE layers are ``artlite.moe``'s. The tokenizer and chat template are
``models.tokenizer``'s (GPT-2 BPE, pre-tokenizer ``flint``; GLM-style turns, GLM-4.5's tool calls)."""
