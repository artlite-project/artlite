"""The files of MiniCPM-V 4.7's vision tests: the mmproj and the encoder tensors the goldens dump."""

import os

from artlite.model_config import MODELS_DIR

MMPROJ = os.path.join(MODELS_DIR, "openbmb", "MiniCPM-V-4.7-1B-gguf", "mmproj-model-f16.gguf")
NAMES = ("pos_embed|layer_inp_normed|Qcur|attn_out|ffn_inp_normed|ffn_up_b|ffn_gelu|ffn_out|layer_out|"
         "vit_merger_ds_out|post_ln|merger_out")
