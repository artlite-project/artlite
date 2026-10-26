"""Qwen3.5 / 3.6 / 3.8 on the NPU (GGUF architectures ``qwen35``, ``qwen35moe``: Gated DeltaNet and full-attention
layers): ``qwen`` (the model; images through ``models.minicpmv``), ``tokenizer`` (the file's vocabulary and chat
template); on the decode overlay (``artlite.decode``) the weight image (``image``), the decode design (``decode``, its
GDN and attention cores ``gdn``, ``attn``), its host side (``overlay``), the MTP drafter (``draft``) and the prefill
(``prefill``, its operators ``gdn_prefill``, ``attn_prefill``, ``draft_prefill``)."""
