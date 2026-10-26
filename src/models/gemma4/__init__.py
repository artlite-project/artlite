"""Gemma 4 E4B on the NPU (GGUF architecture ``gemma4``): ``gemma`` (the model), ``host`` (the inputs the host
computes: embeddings, per-layer inputs), ``tokenizer`` (the file's vocabulary and chat template); on the decode overlay
(``artlite.decode``) the weight image (``image``), the decode design (``decode``, its attention ``attn``), its host side
(``overlay``), the drafter (``draft``) and the prefill (``prefill``, its operators ``attn_prefill`` and
``prefill_ple``)."""
