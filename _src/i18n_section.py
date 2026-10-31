# ==== Bilingual UI strings (Persian default) ====
CHANNEL = "AiWithMase"
YT_URL = "https://youtube.com/@AiWithMase"
TG_URL = "https://t.me/AiWithMase"

UI = {"lang": "fa"}


def set_lang(code):
    if code in ("fa", "en"):
        UI["lang"] = code


def t(key, **fmt):
    value = TEXTS.get(UI["lang"], {}).get(key) or TEXTS["en"].get(key) or key
    return value.format(**fmt) if fmt else value


def compare_idle():
    return f'<div class="aiq-cmp-empty">{t("compare.idle")}</div>'


def tr(fa, en):
    """Inline bilingual string for engine messages (Persian by default)."""
    return en if UI["lang"] == "en" else fa


TEXTS = {
    "fa": {
        "eyebrow": "AIWITHMASE · استودیوی ویدیوی هوشمند",
        "hero.badge": "✦ MiniMax H3 × NVIDIA Open-H3",
        "hero.title": "ویدیو بساز، <span class=\"aiq-grad\">تا 2K</span> برسان.",
        "hero.sub": "از متن، تصویر یا یک کلیپ کوتاه؛ ویدیوی سینمایی با صدای هماهنگ. اول یک پیش‌نمایش سریع ۳۸۴p، بعد بازسازی واقعی تا 2K.",
        "hero.chip1": "⚡ پیش‌نمایش ۴ استپ",
        "hero.chip2": "🎞️ ۷۲۰p / ۱۰۸۰p / 2K",
        "hero.chip3": "🔊 صدای استریو ۳۲kHz",
        "hero.chip4": "🖥️ Kaggle · T4 ×2",
        "hero.yt": "یوتیوب",
        "hero.tg": "تلگرام",
        "section.idea": "ایده و صدا",
        "section.output": "خروجی و زمان‌بندی",
        "section.result": "نتیجه",
        "brand": "",
        "status": "⚡ هستهٔ INT8 · DiT موازی روی ۲×T4 · پیش‌نمایش ۴ استپ · بازسازی NVIDIA Open-H3 · ۲۴fps + صدای استریو ۳۲kHz",
        "intro": (
            "💡 **چطور کار می‌کند:** هر ویدیو اول به‌صورت یک **پیش‌نمایش سریع ۳۸۴p** با صدا ساخته می‌شود. "
            "اگر خروجی **۷۲۰p / ۱۰۸۰p / 2K** را انتخاب کنی، همان لحظه از روی پیش‌نمایش بازسازی می‌شود؛ "
            "یا بعداً از فهرست پیش‌نمایش‌ها هرکدام را آپاسکیل کن. ویدیوی کم‌کیفیت خودت را هم می‌توانی "
            "آپلود و بازسازی کنی. کلیپ‌های ۵ ثانیه‌ای بهترین نتیجه را می‌دهند."
        ),
        "prompt.label": "🎬 صحنه / حرکت",
        "prompt.ph": "یک نمای سینمایی از ... (یا پرامپت کامل ساختاریافتهٔ H3 را بچسبان)",
        "dialogue.label": "🗣️ دیالوگ (اختیاری)",
        "dialogue.ph": "دقیقاً جمله‌هایی که شخصیت می‌گوید",
        "language.label": "🌐 زبان دیالوگ",
        "language.info": "فارسی جزو ۱۱ زبان رسمی H3 نیست؛ کیفیت متغیر است. جمله‌های کوتاه بهترین نتیجه را می‌دهند.",
        "sound.label": "🔊 افکت صوتی / صدای محیط (اختیاری)",
        "sound.ph": "مثلاً باران، ترافیک دور، همهمهٔ جمعیت",
        "music.label": "🎵 موسیقی پس‌زمینه (اختیاری)",
        "music.ph": "مثلاً پیانوی آرام، لوفای",
        "output.label": "📺 خروجی نهایی",
        "output.info": "پیش‌نمایش همیشه اول ساخته می‌شود؛ ۷۲۰p / ۱۰۸۰p / 2K از روی آن بازسازی می‌شوند.",
        "preset.label": "💎 کیفیت / سرعت آپاسکیل",
        "preset.info": "کاشی‌های بزرگ‌تر = انسجام بیشتر، کندتر",
        "aspect.label": "📏 نسبت تصویر",
        "duration.label": "⏱️ مدت",
        "seed.label": "🎲 سید (-۱ = تصادفی)",
        "image.acc": "🖼️ تصویر → ویدیو (فریم اول / آخر)",
        "image.start": "فریم اول (اختیاری)",
        "image.end": "فریم آخر (اختیاری)",
        "image.help": (
            "*فقط فریم اول = جان بخشیدن به تصویر. اول + آخر = درون‌یابی بین دو فریم. حرکت را در پرامپت "
            "توضیح بده؛ تصویر ظاهر را تعیین می‌کند. نسبت تصویر «خودکار» از تصویر پیروی می‌کند.*"
        ),
        "upload.acc": "🎞️ آپاسکیل ویدیوی خودت (کم‌کیفیت → ۷۲۰p / ۱۰۸۰p / 2K)",
        "upload.video": "ویدیوی تو",
        "upload.help": (
            "**روش کار:** یک کلیپ آپلود کن، در بخش **🎬 صحنه / حرکت** توضیح بده چه چیزی در آن است، "
            "**📺 خروجی نهایی** را انتخاب کن و دکمه را بزن. **⏱️ مدت** حداکثر طول مصرفی است. کلیپ با "
            "جزئیات تازه در ابعاد جدید بازسازی می‌شود و صدای خودش حفظ می‌شود.\n\n"
            "*بهترین نتیجه روی فوتیج تمیز و ثابت تا ۵۷۶p؛ کلیپ‌های نویزی یا بسیار فشرده بیشتر تغییر می‌کنند.*"
        ),
        "upload.btn": "⬆️ ویدیوی من را آپاسکیل کن",
        "lora.acc": "🎨 LoRAها (استایل / حرکت — روی پیش‌نمایش و آپاسکیل)",
        "lora1": "LoRA ۱",
        "strength1": "شدت ۱",
        "lora2": "LoRA ۲",
        "strength2": "شدت ۲",
        "custom_url": "لینک LoRA دلخواه (فایل .safetensors در هاگینگ‌فیس)",
        "custom_file": "...یا فایل .safetensors را آپلود کن",
        "custom_strength": "شدت LoRA دلخواه",
        "auto_trigger": "کلمات تریگر هر LoRA خودکار به پرامپت اضافه شود",
        "adv.acc": "⚙️ تنظیمات پیشرفته",
        "sampler": "سمپلر پیش‌نمایش",
        "sampler.info": "euler همانی است که PDMD با آن آموزش دیده",
        "shift": "شیفت جریان پیش‌نمایش (ویدیو)",
        "custom_steps": "استپ‌های پیش‌نمایش (۰ = مقدار PDMD)",
        "upscale_seed": "🎲 سید آپاسکیل (-۱ = سید پیش‌نمایش)",
        "anchor": "فریم اول / آخر در رزولوشن خروجی دوباره لنگر شود",
        "anchor.info": "آپاسکیل «تصویر → ویدیو» را به تصویر تو وفادار نگه می‌دارد",
        "gen": "🎬 بساز",
        "stop": "🛑 توقف",
        "clear": "🗑️ پاک کردن",
        "preview.label": "👀 پیش‌نمایش ۳۸۴p / ویدیوی تو (با صدا)",
        "final.label": "🎥 ویدیوی بازسازی‌شده (با صدا)",
        "compare.title": "🆚 مقایسهٔ قبل / بعد",
        "compare.idle": (
            "🆚 مقایسهٔ قبل / بعد (پیش‌نمایش یا ویدیوی تو در برابر نسخهٔ بازسازی‌شده) بعد از آپاسکیل اینجا "
            "نمایش داده می‌شود."
        ),
        "picker.label": "🎞️ پیش‌نمایش برای آپاسکیل",
        "picker.info": "همهٔ پیش‌نمایش‌ها نگه داشته می‌شوند؛ هرکدام را انتخاب کن",
        "upscale.btn": "⬆️ پیش‌نمایش انتخاب‌شده را به خروجی نهایی آپاسکیل کن",
        "latest.btn": "📂 آخرین ویدیو را بارگذاری کن (اگر اتصال UI قطع شد)",
        "status.label": "ℹ️ وضعیت",
        "footer": "ساخته‌شده با ❤️ توسط <b>AiWithMase</b>",
        "examples": "✨ نمونه پرامپت‌ها",
        "ready": "آماده است.",
        "job.running": "⏳ یک کار در حال اجراست. ویدیوها به‌محض آماده‌شدن همین‌جا ظاهر می‌شوند.",
        "stop.none": "چیزی در حال اجرا نیست.",
        "stop.stopping": "⏹️ بعد از مرحله / کاشی فعلی متوقف می‌شود...",
        "start.gen": (
            "🚀 شروع شد. پیشرفت بالای صفحه نمایش داده می‌شود؛ اول پیش‌نمایش و بعد ویدیوی بازسازی‌شده "
            "می‌آید. رفرش صفحه یا قطع اتصال کار را متوقف نمی‌کند."
        ),
        "start.upscale": "🚀 پیش‌نمایش انتخاب‌شده به {output} آپاسکیل می‌شود. پیشرفت بالای صفحه است.",
        "start.upload": "🚀 ویدیوی تو رمزگذاری و به {output} بازسازی می‌شود. پیشرفت بالای صفحه است.",
        "pick.output": "❌ اول در 📺 خروجی نهایی یکی از ۷۲۰p، ۱۰۸۰p یا 2K را انتخاب کن.",
        "no.preview": "❌ اول یک پیش‌نمایش بساز (🎬 بساز)، بعد اینجا آپاسکیلش کن.",
        "no.video": "❌ اول یک ویدیو آپلود کن.",
        "nothing.yet": "هنوز ویدیویی در {path} نیست.",
        "loaded.preview": "📁 آخرین پیش‌نمایش بارگذاری شد: {path}",
        "loaded.video": "📁 آخرین ویدیو بارگذاری شد: {path}",
        "eta.head": "⏱️ **زمان تقریبی** (Kaggle T4×2): پیش‌نمایش {w}×{h} ≈ {time}",
        "eta.nofit": " · ❌ {output} برای {duration} در حافظهٔ گرافیک جا نمی‌شود؛ مدت کوتاه‌تری انتخاب کن",
        "eta.upscale": " · بازسازی {output} به {w}×{h} ≈ {time} ({tiles} کاشی × {steps} استپ)",
        "picker.nofit": "❌ {output} برای این پیش‌نمایش {duration} ثانیه‌ای در حافظهٔ گرافیک جا نمی‌شود.",
        "picker.eta": "⏱️ آپاسکیل این پیش‌نمایش به {output} ({w}×{h}) ≈ {time} ({tiles} کاشی × {steps} استپ)",
        "tile.one": "کاشی",
        "tile.many": "کاشی",
        "compare.before": "قبل",
        "compare.after": "بعد",
        "compare.play": "▶ پخش",
        "compare.pause": "⏸ مکث",
        "compare.hint": "برای جابه‌جایی خط جداکننده روی ویدیو بکش · Space: پخش / مکث · ← →: یک فریم",
        "compare.prev": "فریم قبلی (←)",
        "compare.next": "فریم بعدی (→)",
        "compare.loop": "تکرار",
        "compare.sound": "صدا",
        "compare.full": "تمام‌صفحه",
        "starting": "در حال شروع…",
        "working": "در حال کار…",
        "compare.yourvideo": "ویدیوی تو",
        "compare.preview": "پیش‌نمایش ۳۸۴p",
        "demo.title": "استودیوی ویدیوی AiWithMase | MiniMax H3 + Open-H3 2K",
    },
    "en": {
        "eyebrow": "AIWITHMASE · AI VIDEO STUDIO",
        "hero.badge": "✦ MiniMax H3 × NVIDIA Open-H3",
        "hero.title": "Make video. <span class=\"aiq-grad\">Regenerate to 2K.</span>",
        "hero.sub": "From text, an image or a short clip: a cinematic video with synchronized sound. A fast 384p preview first, then a real regenerate pass up to 2K.",
        "hero.chip1": "⚡ 4-step preview",
        "hero.chip2": "🎞️ 720p / 1080p / 2K",
        "hero.chip3": "🔊 32 kHz stereo audio",
        "hero.chip4": "🖥️ Kaggle · T4 ×2",
        "hero.yt": "YouTube",
        "hero.tg": "Telegram",
        "section.idea": "Idea & sound",
        "section.output": "Output & timing",
        "section.result": "Result",
        "brand": "",
        "status": "⚡ INT8 tensor cores · DiT tensor-parallel on 2×T4 · 4-step preview · NVIDIA Open-H3 regenerate · 24 fps + 32 kHz stereo audio",
        "intro": (
            "💡 **How it works:** every video starts as a fast **384p preview** with sound. Pick a "
            "**720p / 1080p / 2K** output and it is regenerated from the preview right away, or upscale any "
            "saved preview later. You can also upload your own low-res clip and regenerate it. 5-second clips "
            "are the sweet spot."
        ),
        "prompt.label": "🎬 Scene / action",
        "prompt.ph": "A cinematic shot of ... (or paste a full H3-structured prompt)",
        "dialogue.label": "🗣️ Spoken line (optional)",
        "dialogue.ph": "The exact words the character says",
        "language.label": "🌐 Dialogue language",
        "language.info": "Persian is not one of H3's 11 officially stable languages; quality varies. Short lines work best.",
        "sound.label": "🔊 Sound effects / ambience (optional)",
        "sound.ph": "e.g. rain, distant traffic, a crowd",
        "music.label": "🎵 Background music (optional)",
        "music.ph": "e.g. soft piano, lo-fi",
        "output.label": "📺 Final output",
        "output.info": "The preview is always made first; 720p / 1080p / 2K are regenerated from it.",
        "preset.label": "💎 Upscale quality / speed",
        "preset.info": "Bigger tiles = more global coherence, slower",
        "aspect.label": "📏 Aspect ratio",
        "duration.label": "⏱️ Duration",
        "seed.label": "🎲 Seed (-1 = random)",
        "image.acc": "🖼️ Image → Video (start / end frame)",
        "image.start": "Start frame (optional)",
        "image.end": "End frame (optional)",
        "image.help": (
            "*Start only = animate the image. Start + end = first/last-frame interpolation. Describe the motion "
            "in the prompt; the image fixes the look. Aspect ratio *Auto* follows the image.*"
        ),
        "upload.acc": "🎞️ Upscale your own video (low-res clip → 720p / 1080p / 2K)",
        "upload.video": "Your video",
        "upload.help": (
            "**How:** upload a clip, describe it in **🎬 Scene / action**, pick the **📺 Final output** and press "
            "the button. **⏱️ Duration** is the maximum length used. The clip is re-drawn at the new size with "
            "fresh detail, and its own sound is kept.\n\n"
            "*Works best on clean, steady footage up to 576p; very noisy or heavily compressed clips may change more.*"
        ),
        "upload.btn": "⬆️ Upscale my video",
        "lora.acc": "🎨 LoRAs (style / motion add-ons, used by the preview and the upscale)",
        "lora1": "LoRA 1",
        "strength1": "Strength 1",
        "lora2": "LoRA 2",
        "strength2": "Strength 2",
        "custom_url": "Custom LoRA URL (Hugging Face .safetensors link)",
        "custom_file": "...or upload a .safetensors",
        "custom_strength": "Custom strength",
        "auto_trigger": "Automatically add each LoRA's trigger words to the prompt",
        "adv.acc": "⚙️ Advanced",
        "sampler": "Preview sampler",
        "sampler.info": "euler: what PDMD was trained for",
        "shift": "Preview flow shift (video)",
        "custom_steps": "Preview steps (0 = PDMD's 4)",
        "upscale_seed": "🎲 Upscale seed (-1 = the preview's seed)",
        "anchor": "Re-anchor the start / end image at the output resolution",
        "anchor.info": "Keeps an Image → Video upscale faithful to your picture",
        "gen": "🎬 Generate",
        "stop": "🛑 Stop",
        "clear": "🗑️ Clear",
        "preview.label": "👀 384p preview / your video (with audio)",
        "final.label": "🎥 Upscaled video (with audio)",
        "compare.title": "🆚 Before / after",
        "compare.idle": (
            "🆚 The before / after comparison (preview or your video vs the regenerated video) appears here "
            "after an upscale."
        ),
        "picker.label": "🎞️ Preview to upscale",
        "picker.info": "Every preview is kept; pick any of them",
        "upscale.btn": "⬆️ Upscale selected preview to the 📺 Final output",
        "latest.btn": "📂 Load latest video (use if the UI lost its connection)",
        "status.label": "ℹ️ Status",
        "footer": "Made with ❤️ by <b>AiWithMase</b>",
        "examples": "✨ Example prompts",
        "ready": "Ready.",
        "job.running": "⏳ A job is already running. Its videos appear here as soon as they are ready.",
        "stop.none": "Nothing is running.",
        "stop.stopping": "⏹️ Stopping after the current step / tile...",
        "start.gen": (
            "🚀 Started. Progress is shown at the top of the page; the preview appears first and the upscaled "
            "video after it. Reloading the page or a dropped connection does not stop the job."
        ),
        "start.upscale": "🚀 Upscaling the selected preview to {output}. Progress is shown at the top of the page.",
        "start.upload": "🚀 Encoding your video and upscaling it to {output}. Progress is shown at the top.",
        "pick.output": "❌ Pick 720p, 1080p or 2K in 📺 Final output first.",
        "no.preview": "❌ Make a preview first (🎬 Generate), then upscale it here.",
        "no.video": "❌ Upload a video first.",
        "nothing.yet": "No videos in {path} yet.",
        "loaded.preview": "📁 Loaded latest preview: {path}",
        "loaded.video": "📁 Loaded latest video: {path}",
        "eta.head": "⏱️ **Estimated time** (Kaggle T4×2): {w}×{h} preview ≈ {time}",
        "eta.nofit": " · ❌ {output} for {duration} does not fit the VRAM: pick a shorter duration",
        "eta.upscale": " · {output} {w}×{h} upscale ≈ {time} ({tiles} × {steps} steps)",
        "picker.nofit": "❌ {output} does not fit the VRAM for this {duration}s preview.",
        "picker.eta": "⏱️ Upscaling this preview to {output} ({w}×{h}) ≈ {time} ({tiles} × {steps} steps)",
        "tile.one": "tile",
        "tile.many": "tiles",
        "compare.before": "BEFORE",
        "compare.after": "AFTER",
        "compare.play": "▶ Play",
        "compare.pause": "⏸ Pause",
        "compare.hint": "Drag on the video to move the divider · Space: play / pause · ← →: one frame",
        "compare.prev": "Previous frame (←)",
        "compare.next": "Next frame (→)",
        "compare.loop": "Loop",
        "compare.sound": "Sound",
        "compare.full": "Fullscreen",
        "starting": "Starting…",
        "working": "Working…",
        "compare.yourvideo": "your video",
        "compare.preview": "384p preview",
        "demo.title": "AiWithMase Video Studio | MiniMax H3 + Open-H3 2K",
    },
}
