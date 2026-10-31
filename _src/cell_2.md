## ⚙️ Step 1 — Environment, dependencies and model downloads

Installs the pinned Wan2GP code and its H3 dependencies, then downloads the weights in parallel:

* MiniMax H3 base transformer (INT8 ConvRot), Qwen3-VL-32B text encoder (GGUF), video + audio VAEs and the PDMD 4-step preview LoRA
* **Open-H3-Regenerate-2K**: the latent upscaler, the refiner LoRA (12.45 GB FP32, converted to a 3.1 GB INT8 file while it downloads) and the 50 LMS Sharp curve branches (stored as 155 MB of precomputed matrices)

The preview LoRA and the Open-H3 refiner both work on the same base H3 transformer, so preview and upscale share one copy of the model in VRAM. The preview's soundtrack becomes the final video's audio.