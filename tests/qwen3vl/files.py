"""The files of Qwen3.6-35B-A3B's vision tests."""

import os

from artlite.model_config import MODELS_DIR, qwen_gguf

MODEL = qwen_gguf("Q4_K_M", "35B-A3B")
MMPROJ = os.path.join(MODELS_DIR, "lmstudio-community", "Qwen3.6-35B-A3B-GGUF", "mmproj-Qwen3.6-35B-A3B-BF16.gguf")
