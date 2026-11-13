"""llama.cpp's CPU runs as references: ``runner`` builds the dump tools against the llama.cpp tree and caches their outputs,
``text`` is a token sequence's tensors and logits (any architecture), ``media`` a media prompt's through libmtmd."""
