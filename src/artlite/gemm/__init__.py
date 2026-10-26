"""The prefill GEMMs on the decode weight tiles: ``gemm_q`` (GemmQ, the tile-major path and the planar helpers),
``gemm_qd`` (GemmQd: one dequantizing core per column, the planar gather), ``mac`` (their shared MAC loop)."""
