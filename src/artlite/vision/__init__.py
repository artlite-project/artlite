"""The vision overlay: a whole image network per dispatch (Stable Diffusion's UNet and VAE decoder: ``models.sd15``;
MiniCPM-V's vision encoder: ``models.minicpmv``) on one static design run by host-built transactions (``design``: the
array; ``core``: the cores' program and its roles; ``seq``: the transactions; ``ops``: GEMMs and convs, the norms'
applies, attention; ``layout``: the memtile tensors; ``build``: a dispatch's program and its DDR images; ``layers``:
a network's tensors and layers; ``device``: the design on the NPU, a model's cached state)."""
