"""Strix Halo's numbers from the same diagnosis: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, 8 x
16 GB LPDDR5 8000 MT/s, Linux 7.2.13-gentoo-dist-bin, amdxdna_accel_driver 0.18.0, firmware 1.1.2.65,
platform profile performance, plugged in (artlite-probe --diagnose)."""

HALO: dict[str, dict] = {
    "dma": {
        "2 MB": {
            "DDR read, 1 channel (GB/s)": 13.6,
            "DDR read, 1 shim: 2 channels": 26.5,
            "DDR read, 8 shims: 16 channels": 56.7,
            "DDR read, 8 shims: 1 channel each": 56.4,
            "DDR write, 16 channels": 57.4,
            "DDR read 8 + write 8 channels": 76.1,
            "gather 256 B rows 64 KB apart, 16 channels": 15.5,
            "gather 512 B rows 64 KB apart": 29.6,
            "gather 1 KB rows 64 KB apart": 52.0,
            "gather 256 B rows 64 KB + 256 B apart": 56.7,
            "write 128 B rows 64 KB + 128 B apart, 16 channels": 56.9,
            "write 132 B rows 64 KB + 132 B apart": 41.0,
            "DDR read, 16 channels, regions at other 2 MB-page offsets": 56.7,
            "DDR write, 16 channels, regions at other 2 MB-page offsets": 57.5,
            "DDR read 8 + write 8, regions at other 2 MB-page offsets": 75.4,
            "write 576 B rows 64 KB + 576 B apart (whole lines)": 57.4,
            "write 132 B rows, regions at other 2 MB-page offsets": 41.2,
            "read 2 KB rows 9280 B apart (Q4_K weight slices)": 57.1
        },
        "4 KB": {
            "DDR read, 1 channel (GB/s)": 13.8,
            "DDR read, 1 shim: 2 channels": 26.7,
            "DDR read, 8 shims: 16 channels": 56.7,
            "DDR read, 8 shims: 1 channel each": 55.4,
            "DDR write, 16 channels": 53.5,
            "DDR read 8 + write 8 channels": 73.4,
            "gather 256 B rows 64 KB apart, 16 channels": 14.2,
            "gather 512 B rows 64 KB apart": 28.3,
            "gather 1 KB rows 64 KB apart": 50.8,
            "gather 256 B rows 64 KB + 256 B apart": 29.2,
            "write 128 B rows 64 KB + 128 B apart, 16 channels": 12.2,
            "write 132 B rows 64 KB + 132 B apart": 12.4,
            "DDR read, 16 channels, regions at other 2 MB-page offsets": 56.7,
            "DDR write, 16 channels, regions at other 2 MB-page offsets": 57.4,
            "DDR read 8 + write 8, regions at other 2 MB-page offsets": 75.9,
            "write 576 B rows 64 KB + 576 B apart (whole lines)": 34.0,
            "write 132 B rows, regions at other 2 MB-page offsets": 13.0,
            "read 2 KB rows 9280 B apart (Q4_K weight slices)": 56.8
        }
    },
    "control": {
        "noop (ns per op)": 42.2,
        "noop (ns per byte)": 10.5,
        "write32 (ns per op)": 62.5,
        "write32 (ns per byte)": 2.6,
        "blockwrite 1 KB (ns per op)": 1115,
        "blockwrite 1 KB (ns per byte)": 1.1,
        "64 B DMA task + wait (ns per op)": 2065,
        "64 B DMA tasks of 8 columns + one wait (ns per op)": 610,
        "NOP dispatch, blocking (us)": 48.0,
        "NOP dispatches, 1 queued (per s)": 56785,
        "NOP dispatches, 3 queued (per s)": 53854,
        "array clock (MHz)": 1808,
        "dispatch round trip (us)": 55.2,
        "array clock, 32 cores of MMULs (MHz)": 1808,
        "array clock, 32 cores of MMULs, a busy CPU (MHz)": 1808
    },
    "pp1024": {
        "V0: 2 MB pages, whole lines": 8747,
        "V1: 4 KB pages, whole lines": 8517,
        "V2: 2 MB pages, partial lines": 8568,
        "V3: 4 KB pages, partial lines": 8577,
        "V4: V0 with a busy CPU": 8776
    },
    "kinds": {
        "whole program": 107,
        "AttnOperator": 10.1,
        "GemmQOperator Q4_K 2048x2304": 5.5,
        "GemmQOperator Q4_K 2048x8192": 36.0,
        "GemmQOperator Q4_K 4096x2304": 10.4,
        "GemmQdOperator Q6_K 2048x2304": 15.9,
        "PrefillNormOperator": 10.4,
        "PrepOperator": 3.2,
        "gemm qkv Q4_K 2048x3584": 8.8,
        "gemm qkv Q6_K 2048x3584": 2.2
    },
    "design_switch_us": 77.3,
    "xrt_validate": {
        "exit": 0,
        "gemm_tops": "51.3",
        "latency_us": "55.0",
        "throughput_ops": "94408.0",
        "power_mode": "default"
    }
}
