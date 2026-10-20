import numpy as np
import torch

from artlite.operators import act_lut

from .elementwise import SiluOperator, emulate


class TestRtcSiLU:
    def test_silu_exact(self):
        """al::silu (the qwen35 decode GDN core's SiLU) over every finite bf16 in [-30, 30] (but the tiny ones
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

    def test_silu_lut(self):
        """The table SiLU (act_lut.py) over every finite bf16 >= -24 (but the tiny ones whose SiLU is denormal):
        bit-exact against its emulation, within 0.59 bf16 ULPs and 99.8% correctly rounded; below -24
        within 1e-9 of the exact value."""
        x = torch.from_numpy(act_lut.bf16_values()).bfloat16()
        x = x[(x.float().abs() >= 1e-30) | (x == 0)]
        x = torch.cat([x, torch.zeros(-len(x) % 1024, dtype=torch.bfloat16)])
        op = SiluOperator(size=len(x), num_columns=1, tile_size=1024, lut=True).compile()
        out = op.run(x).float().numpy()
        xf = x.float().numpy()
        assert np.array_equal(out, emulate(act_lut.SILU, xf))
        ref = act_lut.silu64(xf)
        m = xf >= -act_lut.SILU.t_max
        err = (np.abs(out[m] - ref[m]) / act_lut.ulp_bf16(ref[m])).max()
        exact = np.mean(out[m] == torch.from_numpy(ref[m]).float().bfloat16().float().numpy())
        assert err <= 0.59 and exact >= 0.998, f"max error {err:.3f} bf16 ULPs, {exact:.2%} correctly rounded"
        assert np.abs(out[~m] - ref[~m]).max() < 1e-9

    def test_sigmoid_lut(self):
        """The table sigmoid (act_lut.SIGMOID: the function over [-16, 16]) over every finite bf16 (but the tiny
        ones): bit-exact against its emulation, within 0.6 bf16 ULP in [-16, 16] (beyond: within 1.2e-7)."""
        x = torch.from_numpy(act_lut.bf16_values()).bfloat16()
        x = x[(x.float().abs() >= 1e-30) | (x == 0)]
        x = torch.cat([x, torch.zeros(-len(x) % 1024, dtype=torch.bfloat16)])
        op = SiluOperator(size=len(x), num_columns=1, tile_size=1024, lut=True, table="sigmoid").compile()
        out = op.run(x).float().numpy()
        xf = x.float().numpy()
        assert np.array_equal(out, emulate(act_lut.SIGMOID, xf))
        ref = act_lut.sigmoid64(xf)
        m = np.abs(xf) <= act_lut.SIGMOID.t_max
        err = (np.abs(out[m] - ref[m]) / act_lut.ulp_bf16(ref[m])).max()
        exact = np.mean(out[m] == torch.from_numpy(ref[m]).float().bfloat16().float().numpy())
        print(f"sigmoid table: max {err:.3f} bf16 ULPs, {exact:.2%} correctly rounded")
        assert err <= 0.6 and exact >= 0.99, f"max error {err:.3f} bf16 ULPs, {exact:.2%} correctly rounded"
        assert np.abs(out[~m] - ref[~m]).max() < 1.2e-7
