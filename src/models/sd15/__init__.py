"""Stable Diffusion 1.5, text to image, from a diffusers directory (``model_index.json``: ``StableDiffusionPipeline``):
``pipeline`` (``StableDiffusion.txt2img`` and the ``Backend`` whose parts run the networks), ``tokenizer`` (CLIP
BPE), ``sampling`` (the noise schedule, the samplers, classifier-free guidance), ``host`` (the UNet's per-step time
biases), ``reference`` (the networks in fp32 torch: the CPU backend), ``npu`` (the UNet and the VAE decoder on the
NPU), ``weights`` (safetensors), ``png``; on the vision overlay (``artlite.vision``) the UNet (``unet``: its blocks,
``level64``: its 64x64 level, ``model``: the step) and the VAE decoder (``vae``)."""
