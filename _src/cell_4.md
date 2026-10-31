## 🚀 Step 2 — Engine + Gradio UI

Writes the engine (`run_h3_2k.py`) and starts it: the models load in about 4 minutes, then the public links are printed.

**In the UI**
* **📺 Final output**: *384p preview only*, **720p** (default), **1080p** or **2K**. The preview is always made first and shown as soon as it is ready
* **💎 Upscale quality / speed**: ⚡ Fast (2 steps, default) · ⚖️ Balanced (3 steps, NVIDIA's setting) · 💎 Quality (3 steps, bigger tiles)
* **🎞️ Preview to upscale** + **⬆️ Upscale selected preview**: every preview is saved with its latent and prompt, so you can upscale any of them later, to any size
* **🎞️ Upscale your own video**: upload a low-res clip (best up to 576p, steady footage), describe it in the prompt, pick the output size. **⏱️ Duration** caps the length used; the clip keeps its own sound
* **🆚 Before / after**: after each upscale, drag the divider to compare the preview (or your video) with the regenerated video (both play in sync; ← → step one frame, ⛶ fullscreen)
* **🎨 LoRAs**: style / motion add-ons with trigger words, applied to the preview and the upscale
* Long jobs run in the background: a reloaded page or a dropped connection does not stop them, and **📂 Load latest video** fetches the newest result