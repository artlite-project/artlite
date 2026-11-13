"""The prefill GEMMs on the decode weight tiles: ``gemm_q`` (GemmQ, the tile-major path) and ``gemm_qd`` (GemmQd: one
dequantizing core per column, the planar gather), over ``geometry`` (what a GEMM's shape makes of its passes, tiles and
C objects), ``dataflow`` (its shim taps and memtile walks), ``pacing`` and ``mac`` (their shared MAC loop)."""
