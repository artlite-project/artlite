# artlite-probe

Measures the NPU's own numbers, next to those of the Strix Halo that Artlite is measured on and of a Krackan Point
laptop: the array clock, the dispatch round trip and the DRAM read and write rates of its DMA. It takes about a minute
and needs no model file.
Model speeds alone cannot tell a slower clock from a narrower memory path, so a report of another machine's speeds
([Share your results](https://github.com/artlite-project/artlite/issues/5)) starts with this table.

```console
$ artlite-probe
artlite-probe
machine: ASUSTeK COMPUTER INC. ROG Flow Z13 GZ302EA_GZ302EA 1.0; AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, 32 CPUs; 121.5 GB of memory
OS: Linux 7.2.9-gentoo-dist-bin; amdxdna_accel_driver 0.18.0 (xdna-driver), firmware 1.1.2.65
power: platform profile performance (performance), plugged in
NPU: power mode default; interrupts on CPU 0 (max 5187 MHz); device heap: grows

| | this machine | Strix Halo | Krackan Point |
|---|---:|---:|---:|
| array clock, measured (MHz) | 1810 | 1810 | 1808 |
| dispatch round trip (us) | 49 | 49 | 56 |
| DDR read, 1 channel (GB/s) | 13.9 | 13.9 | 14.2 |
| DDR read, 1 shim: 2 channels | 26.9 | 26.9 | 28.1 |
| DDR read, 8 shims: 16 channels | 56.8 | 56.8 | 97.8 |
| DDR read, 8 shims: 1 channel each | 56.4 | 56.4 | ? |
| DDR write, 16 channels | 57.6 | 57.6 | 87.2 |
| DDR read 8 + write 8 channels | 76.1 | 76.1 | 88.2 |
| gather 256 B rows 64 KB apart, 16 channels | 15.6 | 15.6 | 28.6 |
| gather 512 B rows 64 KB apart | 30.8 | 30.8 | 55.9 |
| gather 1 KB rows 64 KB apart | 56.6 | 56.6 | 95.6 |
| gather 256 B rows 64 KB + 256 B apart | 56.9 | 56.9 | 89.6 |
| 16-channel read per array cycle (B) | 31.4 | 31.4 | 54.1 |

build: 391fc8d2d (109)
```

| row | what it measures |
|---|---|
| array clock | one core spins until its timer has advanced by a given count of cycles; two counts timed on the host, so the dispatch cancels out |
| dispatch round trip | a dispatch of almost no work, from submit to its completion seen on the host |
| DDR read / write | shim DMA channels streaming DRAM to the memory tiles (or back) on a design with no cores: one channel, one column's two, all 16, one per column (as the decode streams its weights); a read and a write stream in every column at once |
| gather | 16 channels reading rows of 256 B to 1 KB, 64 KB apart: rows of 256 B or less at a stride of a multiple of 4 KB fall on one DRAM channel, a skewed stride spreads them |
| per array cycle | the 16-channel read rate over the measured clock: Strix Halo's NPU moves at most 32 B per array cycle to DRAM |

The reference columns: Strix Halo is a Ryzen AI Max+ 395 (256-bit LPDDR5X-8000, 256 GB/s; AMD's xdna-driver), Krackan
Point a Ryzen AI 7 350 laptop (128-bit LPDDR5X-7500, 120 GB/s; Linux 7.0's `amdxdna`), both plugged in, NPU power mode
`default`. Their arrays are the same, at the same clock, with the same rates per DMA channel and per shim; what differs
is the path to DRAM. Strix Halo's NPU reaches it through a port of 32 B per array cycle (its rate follows the array
clock in every power mode), 22% of that bus; Krackan Point's reads 81% of its bus, 1.7x as much. Small rows at a stride
of a multiple of 4 KB cost both alike, about 70% below a stream at 256 B and 45% at 512 B, so a layout that is fast on
one is fast on the other.

The first lines name the machine, the kernel, which `amdxdna` runs (`amdxdna_accel_driver` 0.6 / 0.8 / 0.10 are Linux
7.0 / 7.1 / 7.2's own), the NPU firmware, the platform profile and the power adapter, the NPU's power mode, the CPU
that takes the NPU's interrupts, and whether the device heap can grow (Linux before 7.2: one 64 MB heap per process).

| option | |
|---|---|
| `--seconds S` | seconds of dispatches per measurement (default 1) |
