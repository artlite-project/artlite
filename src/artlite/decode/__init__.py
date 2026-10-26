"""The decode overlay of the language models (``models.llama``, ``models.gemma4``, ``models.qwen35``): one design whose
cores interpret their inputs, run by host-built transactions (``artlite.overlay``): ``design`` (its operator, the
metadata words), ``kernels`` and ``gemv`` (the main and helper cores' kernels), ``phases`` (a GEMV phase's K groups),
``image`` (the weight image), ``kv`` and ``vring`` (the KV caches), ``attn`` (the attention cores of Gemma and Llama),
``model`` (the host side the models share)."""
