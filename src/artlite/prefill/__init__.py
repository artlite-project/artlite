"""The prefill shared by the language models, on their decode overlay's buffers: ``norm`` (RMSNorm with the residual
add, bfp16 panel output), ``gemm`` (the GEMMs on the decode weight image), ``chunks`` (the chunk programs,
``ChunkPrefill``), ``attn`` (the attention of a column per KV head: Llama's)."""
