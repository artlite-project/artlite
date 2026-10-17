"""Stable Diffusion 1.5, text to image, from a diffusers directory (``model_index.json``: ``StableDiffusionPipeline``):
``pipeline`` (``StableDiffusion.txt2img`` and the ``Backend`` whose parts run the networks), ``tokenizer`` (CLIP
BPE), ``sampling`` (the noise schedule, the samplers, classifier-free guidance), ``host`` (the UNet's per-step time
biases), ``reference`` (the networks in fp32 torch: the CPU backend), ``npu`` (the UNet on the NPU), ``weights``
(safetensors), ``png``."""
