### What does it do?

**MiniMax H3** generates video **and** native 32 kHz stereo audio together: speech with lip sync, sound effects and music, from text, a start image, or first + last frames. **[Open-H3-Regenerate-2K](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/)** (NVIDIA) is an open regenerate pass for H3: it takes a low-resolution video, upscales it in latent space and re-draws it at high resolution in just 2-3 denoising steps, adding real detail instead of blurry upscaling.

This notebook runs both on free Kaggle **GPU T4 x2**:

1. **🎬 Preview**: a 384p video with sound in about 2 minutes, so you can judge motion, framing and audio quickly
2. **🔍 Upscale**: the preview is regenerated at **720p**, **1080p** or **2K (2560×1440)**, keeping its motion, composition and soundtrack
3. **🎞️ Your own video**: upload a low-res clip and regenerate it at 720p, 1080p or 2K the same way, with its own sound

| Feature | Details |
|----|----|
| Preview model | MiniMax H3 base transformer (pruned 20B, INT8 ConvRot) + the [PDMD](https://github.com/ZeamoxWang/pdmd) 4-step distillation LoRA, 672×384 for 16:9 |
| Upscale | [Open-H3-Regenerate-2K](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K): 3D-conv latent upscaler → re-noise → 2-3 Euler steps with the merged refiner LoRA (refiner + LightX2V FL2VA Turbo + LMS Sharp, rank 576) and Sharp's AdaLN curve branches |
| Outputs | 720p (1280×720), 1080p (1920×1080), 2K (2560×1440) for 16:9; portrait, square, 4:3 and 21:9 too |
| Speed | Comfy Kitchen INT8 tensor-core GEMMs · **tensor-parallel DiT** (every block split across both T4s) · LoRAs live as INT8 factors on both GPUs · the upscale runs as overlapping **spatial tiles** blended every step · two-GPU VAE decode streamed straight into the MP4 |
| Workflow | Generate → preview appears → upscale right away (*📺 Final output*) or later from the **🎞️ Preview to upscale** list (every preview is kept) |
| Compare | **🆚 Before / after** player: drag a divider across the preview (or your video) and the regenerated video, frame-accurate and in sync |
| Modes | Text → Video, Image → Video, First + Last Frame → Video (the images are re-anchored at the output resolution), your own video → upscale |
| LoRAs | 9 popular H3 style / motion LoRAs in a dropdown + any custom `.safetensors` URL / upload, used by the preview and the upscale |
| Output | MP4 with audio in `/kaggle/working/outputs` (Kaggle Output panel) |

### Quick Start
1. **Settings → Accelerator → GPU T4 x2**, and **turn on Internet** in the Settings sidebar
2. Run all cells. Step 1 installs and downloads ~55 GB (about 6-12 minutes; the Open-H3 LoRA is converted to INT8 while it downloads, so ~44 GB end up on disk). Step 2 loads the models (about 4 minutes) and starts the UI
3. Open the **Cloudflare** link (live progress bar) or the **Gradio** share link printed at the bottom of Step 2
4. Write a prompt, pick **📺 Final output** (720p by default; 1080p / 2K, or *preview only*) and press **🎬 Generate**. The preview shows first, the upscaled video after it
5. Like a preview but want another size? Pick it in **🎞️ Preview to upscale** and press **⬆️ Upscale selected preview**
6. To upscale **your own clip**: open **🎞️ Upscale your own video**, upload it, describe it in **🎬 Scene / action** and press **⬆️ Upscale my video**

### ⏱️ Typical times (5 s clip, 16:9)
| Output | ⚡ Fast (default) | ⚖️ Balanced | 💎 Quality |
|---|---|---|---|
| 384p preview | ≈ 2 min | — | — |
| 720p | ≈ 5 min | ≈ 8 min | ≈ 8 min |
| 1080p | ≈ 13 min | ≈ 23 min | ≈ 23 min |
| 2K (1440p) | ≈ 23 min | ≈ 41 min | ≈ 46 min |

⚡ **Fast** regenerates in 2 steps with slim tile overlaps: the quickest, with slightly softer fine texture (fur, foliage). ⚖️ **Balanced** is NVIDIA's 3-step setting; 💎 **Quality** adds bigger tiles for more global coherence. Upscale time grows roughly with the **square of the length**, so 5 s clips are the sweet spot. The UI shows an estimate before you start. Your own video adds about a minute for encoding.

> 💡 **Save the downloads:** after the first run, create a Kaggle Dataset from `/kaggle/tmp/h3_models`, `/kaggle/tmp/h3_loras` and `/kaggle/tmp/h3_regen` and attach it with **Add Input**. Step 1 finds the files there and skips the downloads.