"""The prefill of an LFM3 model (in progress: the GEMM plans of its phases, ``plan_gemm``)."""

from artlite.prefill import gemm as PG

DOWN_CHUNKS = 4  # K chunks per down part (2 parts of 4 of the ffn's 8: each part's A strip stays in the memtile, each
# sums 64 k-tiles in its bf16 C tile; the norm adds the parts in fp32)


def plan_gemm(img: object, region: object, T: int, *, pitch: int = 0, ops: dict[str, object] | None = None
              ) -> PG.PhasePlan:
    """A GEMM phase of the prefill (``prefill.gemm.plan_phase``): gate / up with the SiLU epilogue, down in K parts of
    ``DOWN_CHUNKS`` chunks."""
    return PG.plan_phase(img, region, T, pitch=pitch, ops=ops, act="silu",
                         k_part=DOWN_CHUNKS if region.name == "down" else 0)
