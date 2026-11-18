"""The files of MiniCPM-V 4.7's vision tests: the mmproj and the encoder tensors the golden dumps."""

import os

from artlite.model_config import MODELS_DIR

MMPROJ = os.path.join(MODELS_DIR, "openbmb", "MiniCPM-V-4.7-1B-gguf", "mmproj-model-f16.gguf")
NAMES = "post_ln"
