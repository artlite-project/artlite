import torch

from .elementwise import SiluOperator


class TestRtcSiLU:
    def test_silu_exact(self):
        """kmath.silu (the qwen35 decode GDN core's SiLU) over every finite bf16 in [-30, 30] (but the tiny ones
        whose SiLU is denormal: flushed): 98.7% correctly rounded, within 1.17 bf16 ULPs (measured; the
        excess on the negative side is exp(-|x|) at bf16 precision)."""
        x = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        x = x[torch.isfinite(x) & (x.float().abs() <= 30) & ((x.float().abs() >= 1e-30) | (x == 0))]
        x = torch.cat([x, torch.zeros(-len(x) % 1024, dtype=torch.bfloat16)])
        op = SiluOperator(size=len(x), num_columns=1, tile_size=1024).compile()
        out = op.run(x).float()
        ref = torch.nn.functional.silu(x.double())
        ulp = (ref.abs().clamp(min=2.0**-126).log2().floor() - 7).exp2()  # bf16 ULP at the reference
        err = ((out.double() - ref).abs() / ulp).max().item()
        exact = (out == ref.float().bfloat16().float()).double().mean().item()
        assert err <= 1.2 and exact >= 0.985, f"max error {err:.2f} bf16 ULPs, {exact:.2%} correctly rounded"
