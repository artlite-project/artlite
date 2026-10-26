"""Mixture-of-experts layers, model-independent: on the decode overlay ``route`` (the routing on the cores, the control
packets that point the column's weight stream at its experts); in the prefill ``prefill`` (an MoE block of a chunk:
``gemm``, the router and the shared expert; ``plan``, the host's routing and the experts' chunks; ``expert_gemm`` and
``gather``, the work-list operators; ``ctl``, their control streams)."""
