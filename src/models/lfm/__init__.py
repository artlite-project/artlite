"""LiquidAI LFM3 on the NPU (GGUF architecture ``lfm3``): ``lfm`` (the model); on the decode overlay
(``artlite.decode``) the weight image (``image``), the decode design (``decode``) with its conv core (``conv``), its host
side (``overlay``), the prompt's chunk programs (``prefill``) and their conv operator (``conv_prefill``). The tokenizer
and chat template are ``models.tokenizer``'s (GPT-2 BPE, pre-tokenizer ``lfm2``; ChatML with the BOS and an opened
``<think>`` in the template)."""
