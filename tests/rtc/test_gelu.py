import torch

from .elementwise import GeluOperator


def _tanh_ref(x: torch.Tensor) -> torch.Tensor:
    """The tanh form in f64, written as x sigmoid(z) (1 + tanh cancels below x = -6.5 even in f64)."""
    xd = x.double()
    return xd * torch.sigmoid(2 * (2 / torch.pi) ** 0.5 * (xd + 0.044715 * xd**3))


def _domain() -> torch.Tensor:
    """Every finite bf16 in [-12, 12] (but the tiny ones whose GELU is denormal: flushed), padded to 1024s."""
    x = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    x = x[torch.isfinite(x) & (x.float().abs() <= 12) & ((x.float().abs() >= 1e-30) | (x == 0))]
    return torch.cat([x, torch.zeros(-len(x) % 1024, dtype=torch.bfloat16)])


class TestRtcGELU:
    def test_gelu_lut(self):
        """kmath.gelu (the Gemma kernels' activation) over every finite bf16 in [-12, 12] against the tanh form
        in f64. Measured: 97.8% correctly rounded, mean 0.041 ULPs, within 1.7 ULPs everywhere."""
        x = _domain()
        op = GeluOperator(size=len(x), num_columns=1, tile_size=1024).compile()
        out = op.run(x).float()
        ref = _tanh_ref(x)
        keep = ref.abs() >= 2.0**-100  # below: the result underflows towards the denormals
        ulp = (ref.abs().clamp(min=2.0**-126).log2().floor() - 7).exp2()  # bf16 ULP at the reference
        e = ((out.double() - ref).abs() / ulp)[keep]
        exact = (out == ref.float().bfloat16().float())[keep].double().mean().item()
        big = ref[keep].abs() >= 2.0**-6  # (smaller values vanish in a bfp16 block of O(1) activations)
        ab = (out.double() - ref).abs().max().item()
        print(f"\ngelu_lut: mean {e.mean().item():.3f} ULPs, {exact:.2%} correctly rounded; max {e[big].max().item():.2f} "
              f"ULPs where |gelu| >= 2^-6, {e.max().item():.1f} anywhere; max abs error {ab:.2e}")
        assert e.max().item() <= 2 and exact >= 0.97 and e.mean().item() <= 0.06
