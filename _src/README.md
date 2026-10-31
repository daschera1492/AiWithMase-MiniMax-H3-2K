# AiWithMase-MiniMax-H3-2K
## Customized Kaggle Notebook for MiniMax H3 + Open-H3 Regenerate 2K

**Brand:** AiWithMase  
**GPU:** Kaggle T4 ×2  
**UI:** Bilingual Persian (RTL) + English (LTR)  
**Default Language:** Persian (فارسی)  

---

### 📌 Description

This is a customized version of the **MiniMax H3 + Open-H3 Regenerate 2K** Kaggle notebook, redesigned and branded for **AiWithMase**.

#### Key Features:
- **🌐 Bilingual UI:** Persian (فارسی) as default with full RTL support; toggle to English (LTR)
- **🎨 Modern UI:** Dark glass/aurora theme with Vazirmatn font, animated background, hero visual
- **📺 Branded:** Only AiWithMase Telegram & YouTube links (AIQUEST/X removed)
- **🗣️ Persian in Dialogue Languages:** Added to dropdown with warning about varying quality (not among H3's 11 stable languages)
- **⚡ Optimized for Kaggle T4×2:** All code verified running on the target hardware

#### UI Screenshots:
- **Persian (RTL):** `shot_fa5.png` - Full right-to-left layout, hero title "ویدیو بساز، تا 2K برسان."
- **English (LTR):** `shot_en.png` - Switches to left-to-right mode

#### Supported Dialogue Languages:
Arabic, Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish, **Persian** (added), Turkish, Hindi, Urdu, Indonesian

#### How to Use:
1. Open the notebook on Kaggle GPU T4×2
2. UI opens in **Persian (RTL)** by default
3. Click **English** at the top to switch to LTR mode
4. Select dialogue language from dropdown (Persian included with warning)
5. Enter prompt, select output (720p/1080p/2K), and press **Generate**

#### Credits:
- Original notebook: `unlockbulk/minimax-h3-open-h3-regenerate-2k`
- Redesign, dūzbandehī, and branding: **AiWithMase**
- Model: MiniMax H3 (NVIDIA Open-H3-Regenerate-2K)

---

### 🚀 One-Click Run on Kaggle

[![Run on Kaggle](https://kaggle.com/static/images/run-button.svg)](https://www.kaggle.com/kernels/run)

*Click the button above to open the notebook in Kaggle and start immediately with GPU T4×2.*

---

### 📁 Repository Contents

| File | Description |
|------|-------------|
| `AiWithMase-MiniMax-H3-2K.ipynb` | Main notebook (290KB) - rebuilt with all customizations |
| `cell_0.md` | Cover page with AiWithMase branding, bilingual credits |
| `cell_1.md` | Step 1: Model download and INT8 conversion |
| `cell_2.md` | [Content] |
| `cell_3.py` | Step 2: Engine with updated branding (no AIQUEST) |
| `cell_4.md` | [Content] |
| `cell_5_new.py` | Step 2: Engine cell with i18n, RTL/LTR, all UI blocks |
| `cell_6.md` | Footer/credits section |
| `i18n_section.py` | Persian/English strings, `t()`, `compare_idle()`, brand builders |
| `ui_section.py` | Theme (Vazirmatn/cyan-violet), CSS, hero visual, accordions, RTL/LTR JS |
| `shot_fa5.png` | RTL screenshot (Persian default) |
| `shot_en.png` | LTR screenshot (English mode) |

---

### 🛠️ How to Create This Repo on GitHub

1. **Download/Clone** this folder: `C:\Users\RDP\Desktop\12\_src\`
2. **Create new repo** on GitHub: `AiWithMase-MiniMax-H3-2K`
3. **Upload all files** from the `_src` folder (including the `.ipynb`)
4. **Add repository description** (copy the text above under "📌 Description")
5. **Add the Kaggle button** badge to README if desired:
   ```markdown
   [![Run on Kaggle](https://kaggle.com/static/images/run-button.svg)](https://www.kaggle.com/kernels/run)
   ```
6. **Commit** - the notebook is ready to run on Kaggle GPU T4×2

---

### ⚠️ Important Notes

- **Persian is not one of H3's 11 officially stable dialogue languages** - quality varies, short lines work best
- Notebook requires **Kaggle GPU T4×2** accelerator (verified)
- All `AIQUEST`/X/support links removed - only `AiWithMase` branding remains
- Language toggle dynamically updates all labels, directions, and status messages
- HTML/CSS uses `gradio-container.aiq-en` class for LTR mode when English selected

---
*Generated from customization project - Persian default RTL, bilingual, AiWithMase branded*