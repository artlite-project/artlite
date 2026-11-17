"""Spark-X's prompt prefill on the decode overlay's buffers: the GEMMs on the decode's weight image
(``prefill.gemm.plan_phase``, ``plan_gemm``: the input projection's q, k, v and gate rows per column; gate / up with
GELU's erf form in the epilogue; the output projections into the residual, the down in K parts)."""

from artlite.prefill import gemm as PG

DOWN_PARTS = 2  # (the down projection's K parts where its chunk groups split evenly: each part's bf16 C tile sums
# half the k-tiles, the residual add sums the parts in fp32: the 4B's 160 k-tiles as 2 x 80)


def plan_gemm(img: object, region: object, T: int, *, pitch: int = 0, ops: dict[tuple, object] | None = None,
              k_parts: dict[tuple, int] | None = None) -> PG.PhasePlan:
    """A GEMM phase of the prefill (``prefill.gemm.plan_phase``): gate / up with GELU's erf form; the down in
    ``DOWN_PARTS`` K parts of whole chunk groups where they divide (the 4B: 2 x 5 chunks), an output projection whose A
    strips would be refetched per pass in the parts whose strips stay in the memtile (``prefill.gemm.resident_k_part``).
    ``k_parts``: the choices per region shape, kept across calls."""
    k = 0
    if region.name == "down" and region.kc_group > 1 and region.n_kc == DOWN_PARTS * region.kc_group:
        k = region.kc_group
    if region.name in PG.OUT:
        cache = {} if k_parts is None else k_parts
        key = (region.name, region.K, region.kc_group, tuple(b.fmt for b in region.blocks), T, pitch, k)
        if key not in cache:
            cache[key] = PG.resident_k_part(img, region, T, pitch=pitch, k_part=k)
        k = cache[key]
    return PG.plan_phase(img, region, T, pitch=pitch, ops=ops, act="gelu_erf", k_part=k)
