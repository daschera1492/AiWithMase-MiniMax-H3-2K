# ==== Gradio UI (AiWithMase) ====
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Vazirmatn"), "Inter", "Tahoma", "sans-serif"],
).set(
    body_background_fill="#05070f", body_background_fill_dark="#05070f",
    body_text_color="#e6ecfb", body_text_color_dark="#e6ecfb",
    body_text_color_subdued="#94a3b8", body_text_color_subdued_dark="#94a3b8",
    background_fill_primary="#0a0f1e", background_fill_primary_dark="#0a0f1e",
    background_fill_secondary="#0d1426", background_fill_secondary_dark="#0d1426",
    block_background_fill="rgba(15,23,42,0.66)", block_background_fill_dark="rgba(15,23,42,0.66)",
    block_border_color="rgba(148,163,184,0.16)", block_border_color_dark="rgba(148,163,184,0.16)",
    block_label_background_fill="rgba(15,23,42,0)", block_label_background_fill_dark="rgba(15,23,42,0)",
    block_label_text_color="#c7d2fe", block_label_text_color_dark="#c7d2fe",
    block_title_text_color="#e6ecfb", block_title_text_color_dark="#e6ecfb",
    block_info_text_color="#94a3b8", block_info_text_color_dark="#94a3b8",
    input_background_fill="rgba(2,6,23,0.72)", input_background_fill_dark="rgba(2,6,23,0.72)",
    input_background_fill_focus="rgba(2,6,23,0.92)", input_background_fill_focus_dark="rgba(2,6,23,0.92)",
    input_border_color="rgba(148,163,184,0.18)", input_border_color_dark="rgba(148,163,184,0.18)",
    input_border_color_focus="#8b5cf6", input_border_color_focus_dark="#8b5cf6",
    input_placeholder_color="#64748b", input_placeholder_color_dark="#64748b",
    button_primary_background_fill="linear-gradient(135deg,#7c3aed,#c026d3 55%,#0891b2)",
    button_primary_background_fill_dark="linear-gradient(135deg,#7c3aed,#c026d3 55%,#0891b2)",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="rgba(30,41,59,0.86)", button_secondary_background_fill_dark="rgba(30,41,59,0.86)",
    button_secondary_text_color="#e2e8f0", button_secondary_text_color_dark="#e2e8f0",
    checkbox_background_color_selected="#8b5cf6", checkbox_background_color_selected_dark="#8b5cf6",
    slider_color="#8b5cf6", slider_color_dark="#8b5cf6",
    block_radius="18px", container_radius="20px", input_radius="12px",
    block_shadow="0 18px 50px rgba(0,0,0,0.26)", block_shadow_dark="0 18px 50px rgba(0,0,0,0.26)",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800;900&display=swap');
:root { --aiq-violet:#8b5cf6; --aiq-pink:#e879f9; --aiq-cyan:#22d3ee; --aiq-line:rgba(148,163,184,.16); --aiq-muted:#94a3b8; }
html, body { background:#05070f !important; }
.gradio-container {
  max-width:1180px !important; margin:0 auto !important; padding:16px 14px 44px !important;
  background:
    radial-gradient(900px 520px at 8% -6%, rgba(139,92,246,.20), transparent 70%),
    radial-gradient(820px 500px at 100% 0%, rgba(34,211,238,.13), transparent 70%),
    radial-gradient(700px 500px at 50% 105%, rgba(232,121,249,.10), transparent 70%),
    #05070f !important;
  color:#e6ecfb !important;
}
* { font-family:'Vazirmatn','Segoe UI',Tahoma,sans-serif !important; }
#aiq-app { direction:rtl; }
.gradio-container:not(.aiq-en) { direction:rtl; }
.gradio-container.aiq-en { direction:ltr; }
#aiq-deliver { display:none !important; }
.gradio-container:not(.aiq-en) .aiq-hero,
.gradio-container:not(.aiq-en) .aiq-hero h1,
.gradio-container:not(.aiq-en) .aiq-hero p,
.gradio-container:not(.aiq-en) .aiq-chips,
.gradio-container:not(.aiq-en) .aiq-social,
.gradio-container:not(.aiq-en) .aiq-statusbar,
.gradio-container:not(.aiq-en) .aiq-intro,
.gradio-container:not(.aiq-en) .prose,
.gradio-container:not(.aiq-en) .aiq-cmp-empty,
.gradio-container:not(.aiq-en) .aiq-cmp-hint,
.gradio-container:not(.aiq-en) .aiq-footer,
.gradio-container:not(.aiq-en) .aiq-section,
.gradio-container:not(.aiq-en) .aiq-eyebrow-text { direction:rtl !important; text-align:right !important; }
.gradio-container:not(.aiq-en) .block,
.gradio-container:not(.aiq-en) .form,
.gradio-container:not(.aiq-en) label,
.gradio-container:not(.aiq-en) .md,
.gradio-container:not(.aiq-en) .wrap > label,
.gradio-container:not(.aiq-en) [data-testid="block-info"] { direction:rtl !important; }
.gradio-container:not(.aiq-en) .block label,
.gradio-container:not(.aiq-en) .block .label-wrap { text-align:right !important; }
.gradio-container:not(.aiq-en) textarea,
.gradio-container:not(.aiq-en) input[type="text"] { direction:rtl !important; text-align:right !important; }
.gradio-container:not(.aiq-en) input[type="number"] { direction:ltr !important; text-align:right !important; }
.gradio-container:not(.aiq-en) #aiq-status textarea { direction:rtl !important; text-align:right !important; unicode-bidi:plaintext; }
.gradio-container:not(.aiq-en) .wrap-inner { direction:rtl !important; }
.gradio-container:not(.aiq-en) label > span { text-align:right !important; }
.gradio-container .prose { color:#cbd5e1 !important; }
.gradio-container .prose strong { color:#f1f5f9 !important; }
.aiq-top { align-items:center !important; gap:12px !important; margin-bottom:6px !important; }
.aiq-eyebrow-text { color:#8ea0bd; font-size:11.5px; letter-spacing:.16em; text-transform:uppercase; font-weight:700; padding:7px 2px; }
#aiq-lang { min-width:196px; max-width:230px; margin-inline-start:auto; }
#aiq-lang .wrap { display:flex; gap:4px; padding:4px; border-radius:999px; background:rgba(10,15,30,.82) !important; border:1px solid rgba(148,163,184,.18) !important; }
#aiq-lang label { border-radius:999px !important; padding:5px 13px !important; margin:0 !important; border:none !important; background:transparent !important; font-size:12px !important; font-weight:700 !important; color:#a9b6cf !important; }
#aiq-lang label:has(input:checked) { background:linear-gradient(135deg,#7c3aed,#c026d3) !important; color:#fff !important; box-shadow:0 6px 18px rgba(124,58,237,.38) !important; }
#aiq-lang input[type="radio"] { display:none !important; }
.aiq-hero {
  position:relative; overflow:hidden; border-radius:28px; padding:32px 34px 30px;
  background:linear-gradient(135deg, rgba(139,92,246,.22), rgba(10,15,30,.92) 46%, rgba(34,211,238,.13)), rgba(10,15,30,.86);
  border:1px solid rgba(148,163,184,.18);
  box-shadow:0 34px 90px rgba(0,0,0,.48), inset 0 1px 0 rgba(255,255,255,.06);
}
.aiq-hero::after { content:""; position:absolute; width:440px; height:440px; border-radius:50%; top:-180px; right:-140px; background:radial-gradient(circle, rgba(232,121,249,.26), transparent 66%); filter:blur(4px); pointer-events:none; }
.gradio-container.aiq-en .aiq-hero::after { right:auto; left:-140px; }
.aiq-hero > * { position:relative; z-index:1; }
.aiq-visual { position:absolute; left:30px; top:50%; transform:translateY(-50%); width:226px; z-index:2; }
.gradio-container.aiq-en .aiq-visual { left:auto; right:30px; }
.aiq-visual-card { border-radius:18px; padding:9px; background:linear-gradient(160deg, rgba(15,23,42,.92), rgba(2,6,23,.86)); border:1px solid rgba(148,163,184,.22); box-shadow:0 24px 60px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.06); }
.aiq-visual-screen { position:relative; aspect-ratio:16/9; border-radius:12px; overflow:hidden; display:flex; align-items:center; justify-content:center;
  background:radial-gradient(120px 80px at 70% 30%, rgba(232,121,249,.35), transparent 70%), radial-gradient(120px 80px at 30% 70%, rgba(34,211,238,.30), transparent 70%), linear-gradient(135deg,#1e1b4b,#0f172a); }
.aiq-visual-screen::before { content:""; position:absolute; inset:0; background-image:linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px); background-size:22px 22px; }
.aiq-visual-play { position:relative; width:38px; height:38px; border-radius:50%; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.35); display:flex; align-items:center; justify-content:center; color:#fff; font-size:13px; padding-inline-start:2px; backdrop-filter:blur(6px); }
.aiq-visual-meta { display:flex; align-items:center; justify-content:space-between; margin-top:9px; font-size:10px; color:#94a3b8; font-weight:700; }
.aiq-visual-meta b { color:#c4b5fd; }
.aiq-visual-2k { color:#67e8f9 !important; font-weight:800; }
.aiq-visual-bars { display:flex; gap:3px; margin-top:8px; align-items:flex-end; height:16px; }
.aiq-visual-bars i { flex:1; border-radius:2px; background:linear-gradient(180deg,#a78bfa,#22d3ee); opacity:.72; }
.aiq-badge { display:inline-flex; align-items:center; gap:7px; padding:6px 12px; border-radius:999px; font-size:11.5px; font-weight:700; color:#ddd6fe; background:rgba(139,92,246,.15); border:1px solid rgba(139,92,246,.30); }
.aiq-hero h1 { margin:16px 0 10px; font-size:clamp(29px,3.5vw,48px); line-height:1.18; font-weight:800; letter-spacing:-.02em; color:#f8fafc; }
.aiq-grad { background:linear-gradient(90deg,#a78bfa,#e879f9 46%,#67e8f9); -webkit-background-clip:text; background-clip:text; color:transparent; }
.aiq-sub { margin:0; max-width:760px; color:#a9b6cf; font-size:15px; line-height:1.95; }
.aiq-chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }
.aiq-chips span { padding:5px 11px; border-radius:999px; font-size:11.5px; font-weight:600; color:#c7d2fe; background:rgba(148,163,184,.08); border:1px solid rgba(148,163,184,.14); }
.aiq-social { display:flex; flex-wrap:wrap; gap:10px; margin-top:20px; }
.aiq-social-btn { display:inline-flex; align-items:center; gap:8px; padding:10px 17px; border-radius:13px; font-weight:800; font-size:12.5px; color:#fff !important; text-decoration:none !important; border:1px solid rgba(255,255,255,.12); transition:transform .16s ease, box-shadow .16s ease; }
.aiq-social-btn svg { width:18px; height:18px; flex:none; }
.aiq-social-btn.yt { background:linear-gradient(135deg,#ff2448,#b8001f); box-shadow:0 10px 28px rgba(255,36,72,.30); }
.aiq-social-btn.tg { background:linear-gradient(135deg,#38bdf8,#0369a1); box-shadow:0 10px 28px rgba(56,189,248,.26); }
.aiq-social-btn:hover { transform:translateY(-2px); }
.aiq-statusbar { display:flex; align-items:center; gap:9px; flex-wrap:wrap; margin:12px 2px 4px; color:#8ea0bd; font-size:11.5px; }
.aiq-dot { width:7px; height:7px; border-radius:50%; background:#34d399; box-shadow:0 0 12px rgba(52,211,153,.9); flex:none; }
.aiq-intro { color:#b9c6dd !important; font-size:13px !important; line-height:1.9 !important; padding:2px 4px; }
.aiq-card { background:linear-gradient(180deg, rgba(15,23,42,.78), rgba(10,15,30,.72)) !important; border:1px solid rgba(148,163,184,.14) !important; border-radius:22px !important; padding:16px !important; box-shadow:0 18px 50px rgba(0,0,0,.26); }
.aiq-card > .block, .aiq-card .block { box-shadow:none !important; }
.aiq-section { display:flex; align-items:center; gap:9px; margin:6px 2px 8px; color:#c4b5fd; font-size:12.5px; font-weight:800; letter-spacing:.02em; }
.aiq-section::after { content:""; flex:1; height:1px; background:linear-gradient(90deg, rgba(139,92,246,.42), transparent); }
.aiq-acc { border:1px solid rgba(148,163,184,.14) !important; background:rgba(10,15,30,.58) !important; border-radius:18px !important; margin:8px 0 !important; overflow:hidden; }
.aiq-acc > button, .aiq-acc .label-wrap { padding:12px 16px !important; }
.aiq-acc .label-wrap span, .aiq-acc button span { font-weight:800 !important; color:#dbeafe !important; }
textarea, input[type="text"], input[type="number"], input[type="password"] { background:rgba(2,6,23,.72) !important; color:#e2e8f0 !important; border:1px solid rgba(148,163,184,.18) !important; border-radius:12px !important; box-shadow:none !important; }
textarea:focus, input[type="text"]:focus, input[type="number"]:focus { border-color:#8b5cf6 !important; box-shadow:0 0 0 3px rgba(139,92,246,.18) !important; }
.gradio-container label > span { color:#cbd5e1 !important; font-weight:700 !important; }
.gradio-container .info { color:#7f8ea8 !important; font-size:11.5px !important; }
.gradio-container .wrap-inner { background:rgba(2,6,23,.72) !important; border-color:rgba(148,163,184,.18) !important; border-radius:12px !important; }
button { border-radius:12px !important; font-weight:700 !important; }
button.primary { background:linear-gradient(135deg,#7c3aed,#c026d3 55%,#0891b2) !important; border:none !important; color:#fff !important; box-shadow:0 12px 32px rgba(124,58,237,.34) !important; }
button.primary:hover { filter:brightness(1.08); transform:translateY(-1px); }
#stop-btn { background:linear-gradient(135deg,#ef4444,#991b1b) !important; border:none !important; color:#fff !important; }
#clear-btn { background:rgba(30,41,59,.78) !important; border:1px solid rgba(148,163,184,.20) !important; color:#cbd5e1 !important; }
#upload-btn { background:linear-gradient(135deg,#0ea5e9,#0369a1) !important; border:none !important; color:#fff !important; }
#upscale-btn { background:linear-gradient(135deg,#8b5cf6,#0891b2) !important; border:none !important; color:#fff !important; box-shadow:0 10px 28px rgba(139,92,246,.28) !important; }
#gen-btn { min-height:52px; font-size:15px !important; }
#aiq-status textarea { font-size:12.5px !important; line-height:1.85 !important; }
.aiq-videos .block { background:rgba(2,6,23,.58) !important; }
.gradio-container:not(.aiq-en) .aiq-videos label[data-testid="block-label"] { left:auto !important; right:0 !important; }
#aiq-output .wrap { display:grid !important; grid-template-columns:1fr 1fr; gap:8px; }
#aiq-output label { justify-content:center !important; text-align:center !important; padding:9px 6px !important; border-radius:11px !important; border:1px solid rgba(148,163,184,.16) !important; }
#aiq-output label.selected { background:linear-gradient(135deg,#7c3aed,#c026d3 60%,#0891b2) !important; color:#fff !important; border-color:transparent !important; box-shadow:0 10px 24px rgba(124,58,237,.36) !important; }
.aiq-eta { text-align:right; }
.gradio-container.aiq-en .aiq-eta { text-align:left; }
.gradio-container footer { display:none !important; }
.gradio-container .built-with, .gradio-container #footer { display:none !important; }
video { border-radius:13px; background:#000; }
.aiq-footer { text-align:center; color:#7f8ea8; font-size:11.5px; padding:18px 6px 4px; }
.aiq-footer a { color:#a5b4fc !important; text-decoration:none !important; font-weight:700; }
.aiq-footer a:hover { color:#e879f9 !important; }
.aiq-cmp-empty { text-align:center; color:#93a4c0; font-size:12.5px; padding:20px; border:1px dashed rgba(139,92,246,.35); border-radius:16px; background:rgba(10,15,30,.45); }
.aiq-cmp-stage { position:relative; width:100%; max-height:80vh; background:#000; border-radius:16px; overflow:hidden; cursor:ew-resize; touch-action:none; user-select:none; -webkit-user-select:none; }
.aiq-cmp-stage video { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; display:block; pointer-events:none; }
.aiq-cmp-clip { position:absolute; inset:0; clip-path:inset(0 50% 0 0); }
.aiq-cmp-line { position:absolute; top:0; bottom:0; left:50%; width:2px; margin-left:-1px; background:#fff; box-shadow:0 0 8px rgba(0,0,0,.7); pointer-events:none; }
.aiq-cmp-knob { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:38px; height:38px; border-radius:50%; background:linear-gradient(135deg,#8b5cf6,#c026d3); color:#fff; display:flex; align-items:center; justify-content:center; font-size:17px; box-shadow:0 4px 16px rgba(0,0,0,.5); }
.aiq-tag { position:absolute; top:10px; padding:4px 10px; border-radius:7px; font-size:10.5px; font-weight:800; letter-spacing:.05em; color:#fff; background:rgba(0,0,0,.58); pointer-events:none; }
.aiq-tag-l { left:10px; } .aiq-tag-r { right:10px; background:rgba(139,92,246,.88); }
.aiq-cmp-bar { display:flex; align-items:center; gap:8px; margin-top:10px; flex-wrap:wrap; }
.aiq-seek { flex:1; min-width:140px; accent-color:#8b5cf6; }
.aiq-btn { border:none; border-radius:9px; padding:7px 12px; font-weight:700; font-size:12px; cursor:pointer; color:#fff; background:linear-gradient(135deg,#7c3aed,#c026d3); }
.aiq-btn.aiq-off { background:#64748b; }
.aiq-play { min-width:92px; }
.aiq-time { font-size:12px; color:#94a3b8; min-width:92px; text-align:right; font-variant-numeric:tabular-nums; }
.aiq-cmp-hint { font-size:11.5px; color:#7f8ea8; margin-top:6px; text-align:center; }
@media (max-width:1050px) { .aiq-visual { display:none; } }
@media (max-width:860px) { .aiq-grid { flex-direction:column !important; } .aiq-hero { padding:24px 20px; } .aiq-hero h1 { font-size:30px; } .gradio-container { padding:10px 8px 34px !important; } }
"""


RTL_HEAD = """
<script>
(function () {
  window.__aiqLang = window.__aiqLang || 'fa';
  window.__aiqApplyLang = function (lang) {
    window.__aiqLang = (lang === 'en') ? 'en' : 'fa';
    var fa = window.__aiqLang !== 'en';
    var dir = fa ? 'rtl' : 'ltr';
    document.documentElement.setAttribute('dir', dir);
    document.documentElement.lang = fa ? 'fa' : 'en';
    if (document.body) { document.body.setAttribute('dir', dir); document.body.style.direction = dir; }
    var ga = document.querySelector('gradio-app');
    if (ga) { ga.setAttribute('dir', dir); ga.style.direction = dir; }
    var gc = document.querySelector('.gradio-container');
    if (gc) {
      gc.setAttribute('dir', dir);
      gc.style.direction = dir;
      gc.classList.toggle('aiq-en', !fa);
    }
    var app = document.getElementById('aiq-app');
    if (app) { app.setAttribute('dir', dir); app.style.direction = dir; }
  };
  window.__aiqApplyLang(window.__aiqLang);
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { window.__aiqApplyLang(window.__aiqLang); });
  }
  var n = 0;
  var timer = setInterval(function () {
    window.__aiqApplyLang(window.__aiqLang);
    if (++n > 30) { clearInterval(timer); }
  }, 200);
})();
</script>
"""


def brand_html(lang=None):
    lang = lang or UI["lang"]
    tr = lambda k: TEXTS[lang][k]
    return f"""
<div class="aiq-hero">
  <span class="aiq-badge">{tr('hero.badge')}</span>
  <h1>{tr('hero.title')}</h1>
  <p class="aiq-sub">{tr('hero.sub')}</p>
  <div class="aiq-chips"><span>{tr('hero.chip1')}</span><span>{tr('hero.chip2')}</span><span>{tr('hero.chip3')}</span><span>{tr('hero.chip4')}</span></div>
  <div class="aiq-social">
    <a class="aiq-social-btn yt" href="{YT_URL}" target="_blank" rel="noopener">
      <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M23.5 6.2a3 3 0 0 0-2.1-2.1C19.5 3.6 12 3.6 12 3.6s-7.5 0-9.4.5A3 3 0 0 0 .5 6.2 31 31 0 0 0 0 12a31 31 0 0 0 .5 5.8 3 3 0 0 0 2.1 2.1c1.9.5 9.4.5 9.4.5s7.5 0 9.4-.5a3 3 0 0 0 2.1-2.1A31 31 0 0 0 24 12a31 31 0 0 0-.5-5.8z"/><path fill="#0b0f1a" d="M9.6 15.6V8.4L15.9 12z"/></svg>
      {tr('hero.yt')}
    </a>
    <a class="aiq-social-btn tg" href="{TG_URL}" target="_blank" rel="noopener">
      <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M23.9 3.4 20.3 20c-.3 1.2-1 1.5-2 1L12.6 17l-3.1 3c-.3.3-.6.6-1.2.6l.4-5.9L19.5 5.9c.5-.4-.1-.7-.7-.3L5.6 13.9.9 12.4c-1-.3-1-1 .2-1.5L22.3 2.7c.9-.3 1.7.2 1.6.7z"/></svg>
      {tr('hero.tg')}
    </a>
  </div>
  <div class="aiq-visual" aria-hidden="true">
    <div class="aiq-visual-card">
      <div class="aiq-visual-screen"><span class="aiq-visual-play">▶</span></div>
      <div class="aiq-visual-meta" dir="ltr"><span>384p → 720p →</span><span class="aiq-visual-2k">2K</span></div>
      <div class="aiq-visual-bars"><i style="height:30%"></i><i style="height:55%"></i><i style="height:42%"></i><i style="height:70%"></i><i style="height:50%"></i><i style="height:85%"></i><i style="height:62%"></i><i style="height:38%"></i><i style="height:74%"></i><i style="height:46%"></i></div>
    </div>
  </div>
</div>"""


def status_html(lang=None):
    lang = lang or UI["lang"]
    return f'<div class="aiq-statusbar"><span class="aiq-dot"></span><span>{TEXTS[lang]["status"]}</span></div>'


def footer_html(lang=None):
    lang = lang or UI["lang"]
    return (f'<div class="aiq-footer">{TEXTS[lang]["footer"]} &nbsp;·&nbsp; '
            f'<a href="{YT_URL}" target="_blank" rel="noopener">YouTube</a> &nbsp;·&nbsp; '
            f'<a href="{TG_URL}" target="_blank" rel="noopener">Telegram</a></div>')


OUTPUT_LABELS = {
    "fa": {"384p preview only": "پیش‌نمایش ۳۸۴p فقط", "720p": "۷۲۰p", "1080p": "۱۰۸۰p", "2K (1440p)": "2K (۱۴۴۰p)"},
    "en": {"384p preview only": "384p preview only", "720p": "720p", "1080p": "1080p", "2K (1440p)": "2K (1440p)"},
}
PRESET_LABELS = {
    "fa": {"⚡ Fast - 2 steps, slim overlaps": "⚡ سریع — ۲ استپ",
           "⚖️ Balanced - 3 steps": "⚖️ متعادل — ۳ استپ",
           "💎 Quality - 3 steps, bigger tiles": "💎 کیفیت — ۳ استپ، کاشی بزرگ‌تر"},
    "en": {k: k for k in REFINE_PRESETS},
}
ASPECT_LABELS = {
    "fa": {"16:9 Landscape": "۱۶:۹ افقی", "9:16 Portrait": "۹:۱۶ عمودی", "1:1 Square": "۱:۱ مربع",
           "4:3 Classic": "۴:۳ کلاسیک", "3:4 Portrait": "۳:۴ عمودی", "21:9 Cinema": "۲۱:۹ سینمایی",
           "Auto (match image)": "خودکار (مطابق تصویر)"},
    "en": {k: k for k in ASPECTS},
}
LANG_LABELS = {
    "fa": {"English": "انگلیسی", "Chinese": "چینی", "Spanish": "اسپانیایی", "French": "فرانسوی", "German": "آلمانی",
           "Japanese": "ژاپنی", "Korean": "کره‌ای", "Hindi": "هندی", "Urdu": "اردو", "Arabic": "عربی",
           "Portuguese": "پرتغالی", "Russian": "روسی", "Italian": "ایتالیایی", "Turkish": "ترکی",
           "Indonesian": "اندونزیایی", "Persian": "فارسی"},
    "en": {k: k for k in LANGUAGES},
}


def output_choices(lang):
    return [(OUTPUT_LABELS[lang].get(k, k), k) for k in (PREVIEW_ONLY, *OUTPUTS)]


def preset_choices(lang):
    return [(PRESET_LABELS[lang].get(k, k), k) for k in REFINE_PRESETS]


def aspect_choices(lang):
    return [(ASPECT_LABELS[lang].get(k, k), k) for k in ASPECTS]


def language_choices(lang):
    return [(LANG_LABELS[lang].get(k, k), k) for k in LANGUAGES]


DURATION_LABELS = {
    "fa": {"3 s": "۳ ثانیه", "4.5 s": "۴٫۵ ثانیه", "5 s": "۵ ثانیه", "6.5 s": "۶٫۵ ثانیه", "8 s": "۸ ثانیه",
           "10 s": "۱۰ ثانیه", "12 s": "۱۲ ثانیه", "15 s": "۱۵ ثانیه"},
    "en": {k: k for k in DURATIONS},
}


def duration_choices(lang):
    return [(DURATION_LABELS[lang].get(k, k), k) for k in DURATIONS]


I18N = []


def reg(comp, **props):
    I18N.append((comp, props))
    return comp


def switch_lang(lang, output, preset, duration, aspect, picker_name):
    set_lang(lang)
    lang = UI["lang"]
    updates = []
    for comp, props in I18N:
        kw = {}
        for prop, key in props.items():
            if callable(key):
                kw[prop] = key(lang)
            else:
                kw[prop] = TEXTS[lang][key]
        updates.append(gr.update(**kw))
    updates.append(gr.update(value=eta_text(output, preset, duration, aspect)))
    updates.append(gr.update(value=picker_eta(picker_name, output, preset)))
    if not JOB["final_n"]:
        updates.append(gr.update(value=compare_idle()))
    else:
        updates.append(gr.skip())
    return updates


LANG_JS = """
(lang, output, preset, duration, aspect, picker) => {
  if (window.__aiqApplyLang) {
    window.__aiqApplyLang(lang);
  } else {
    const fa = lang === 'fa';
    const container = document.querySelector('.gradio-container');
    if (container) {
      container.classList.toggle('aiq-en', !fa);
      container.setAttribute('dir', fa ? 'rtl' : 'ltr');
    }
    document.documentElement.lang = fa ? 'fa' : 'en';
    document.documentElement.dir = fa ? 'rtl' : 'ltr';
    document.body.style.direction = fa ? 'rtl' : 'ltr';
  }
  return [lang, output, preset, duration, aspect, picker];
}
"""

LORA_CHOICES = [NO_LORA] + list(LORA_CATALOG)
EXAMPLES = [
    ["A street musician plays a glowing violin on a rainy neon-lit Tokyo night, the camera slowly circles him as "
     "puddles reflect pink and blue signs and people with umbrellas pass by.",
     "", "English", "Rain on pavement, distant traffic, the violin melody echoing between buildings.", ""],
    ["A young woman in a cozy cafe looks up from her laptop, smiles warmly at the camera and leans forward.",
     "سلام! خوش اومدی، بهترین صندلی کنار پنجره رو برات نگه داشتم.", "Persian",
     "همهمهٔ آرام کافه، صدای دستگاه اسپرسو، برخورد فنجان‌ها.", "جاز لوفای ملایم."],
    ["A red fox runs across a snowy ridge at sunrise, powder spraying from every stride as a long-lens camera pans with it.",
     "", "English", "Fast paw impacts in powder, cold wind, distant crows.", "Low sustained cellos rising slowly."],
]

print("🖥️ Building the UI...")
sys.stdout.flush()
REFINE_FREE[0] = min(torch.cuda.mem_get_info(d)[0] for d in DEVICES) - REFINE_LORA_BYTES - 100 * 2**20  # for estimates
with gr.Blocks(theme=THEME, css=CSS, head=RTL_HEAD + PROGRESS_HEAD, title=t("demo.title"), elem_id="aiq-app") as demo:
    with gr.Row(elem_classes="aiq-top"):
        with gr.Column(scale=6, min_width=220):
            top = reg(gr.HTML(f'<div class="aiq-eyebrow-text">{t("eyebrow")}</div>'),
                      value=lambda l: f'<div class="aiq-eyebrow-text">{TEXTS[l]["eyebrow"]}</div>')
        with gr.Column(scale=2, min_width=200):
            lang_radio = gr.Radio([("فارسی", "fa"), ("English", "en")], value="fa", show_label=False,
                                  container=False, elem_id="aiq-lang", rtl=True)
    brand = reg(gr.HTML(brand_html()), value=lambda l: brand_html(l))
    status = reg(gr.HTML(status_html()), value=lambda l: status_html(l))
    intro = reg(gr.Markdown(t("intro"), elem_classes="aiq-intro"), value="intro")

    with gr.Row(elem_classes="aiq-grid"):
        with gr.Column(scale=3, elem_classes="aiq-card"):
            reg(gr.HTML(f'<div class="aiq-section">{t("section.idea")}</div>'),
                value=lambda l: f'<div class="aiq-section">{TEXTS[l]["section.idea"]}</div>')
            prompt = reg(gr.Textbox(label=t("prompt.label"), lines=4, placeholder=t("prompt.ph")),
                         label="prompt.label", placeholder="prompt.ph")
            with gr.Row():
                dialogue = reg(gr.Textbox(label=t("dialogue.label"), lines=1, scale=3, placeholder=t("dialogue.ph")),
                               label="dialogue.label", placeholder="dialogue.ph")
                language = reg(gr.Dropdown(language_choices("fa"), value="Persian", label=t("language.label"),
                                           info=t("language.info"), scale=2),
                               label="language.label", info="language.info", choices=lambda l: language_choices(l))
            with gr.Row():
                sound = reg(gr.Textbox(label=t("sound.label"), lines=1, placeholder=t("sound.ph")),
                            label="sound.label", placeholder="sound.ph")
                music = reg(gr.Textbox(label=t("music.label"), lines=1, placeholder=t("music.ph")),
                            label="music.label", placeholder="music.ph")
        with gr.Column(scale=2, elem_classes="aiq-card"):
            reg(gr.HTML(f'<div class="aiq-section">{t("section.output")}</div>'),
                value=lambda l: f'<div class="aiq-section">{TEXTS[l]["section.output"]}</div>')
            output = reg(gr.Radio(output_choices("fa"), value="720p", label=t("output.label"), info=t("output.info"),
                                  elem_id="aiq-output"),
                         label="output.label", info="output.info", choices=lambda l: output_choices(l))
            preset = reg(gr.Dropdown(preset_choices("fa"), value=DEFAULT_PRESET, label=t("preset.label"),
                                     info=t("preset.info")),
                         label="preset.label", info="preset.info", choices=lambda l: preset_choices(l))
            with gr.Row():
                aspect = reg(gr.Dropdown(aspect_choices("fa"), value="16:9 Landscape", label=t("aspect.label")),
                             label="aspect.label", choices=lambda l: aspect_choices(l))
                duration = reg(gr.Dropdown(duration_choices("fa"), value="5 s", label=t("duration.label")),
                               label="duration.label", choices=lambda l: duration_choices(l))
            seed = reg(gr.Number(value=-1, precision=0, label=t("seed.label")), label="seed.label")
    eta = gr.Markdown(eta_text("720p", DEFAULT_PRESET, "5 s", "16:9 Landscape"), elem_classes="aiq-eta")

    with gr.Accordion(t("image.acc"), open=False, elem_classes="aiq-acc") as acc_img:
        reg(acc_img, label="image.acc")
        with gr.Row():
            start_img = reg(gr.Image(type="pil", label=t("image.start"), height=220), label="image.start")
            end_img = reg(gr.Image(type="pil", label=t("image.end"), height=220), label="image.end")
        reg(gr.Markdown(t("image.help")), value="image.help")

    with gr.Accordion(t("upload.acc"), open=False, elem_classes="aiq-acc") as acc_up:
        reg(acc_up, label="upload.acc")
        with gr.Row():
            upload_video = reg(gr.Video(label=t("upload.video"), sources=["upload"], height=260, scale=3),
                               label="upload.video")
            reg(gr.Markdown(t("upload.help")), value="upload.help")
        upload_btn = reg(gr.Button(t("upload.btn"), size="lg", elem_id="upload-btn"), value="upload.btn")

    with gr.Accordion(t("lora.acc"), open=False, elem_classes="aiq-acc") as acc_lora:
        reg(acc_lora, label="lora.acc")
        with gr.Row():
            lora1 = gr.Dropdown(LORA_CHOICES, value=NO_LORA, label=t("lora1"))
            reg(lora1, label="lora1")
            s1 = reg(gr.Slider(0.0, 1.5, value=1.0, step=0.05, label=t("strength1")), label="strength1")
        lora1_info = gr.Markdown("")
        with gr.Row():
            lora2 = gr.Dropdown(LORA_CHOICES, value=NO_LORA, label=t("lora2"))
            reg(lora2, label="lora2")
            s2 = reg(gr.Slider(0.0, 1.5, value=1.0, step=0.05, label=t("strength2")), label="strength2")
        lora2_info = gr.Markdown("")
        with gr.Row():
            custom_url = reg(gr.Textbox(label=t("custom_url"), scale=3,
                                        placeholder="https://huggingface.co/user/repo/resolve/main/my_h3_lora.safetensors"),
                             label="custom_url")
            custom_file = reg(gr.File(label=t("custom_file"), file_types=[".safetensors"], scale=2),
                              label="custom_file")
            s3 = reg(gr.Slider(0.0, 1.5, value=1.0, step=0.05, label=t("custom_strength"), scale=1),
                     label="custom_strength")
        auto_trigger = reg(gr.Checkbox(value=True, label=t("auto_trigger")), label="auto_trigger")

    with gr.Accordion(t("adv.acc"), open=False, elem_classes="aiq-acc") as acc_adv:
        reg(acc_adv, label="adv.acc")
        with gr.Row():
            sampler = reg(gr.Dropdown(["euler", "res_multistep"], value=DEFAULT_SAMPLER, label=t("sampler"),
                                      info=t("sampler.info")), label="sampler", info="sampler.info")
            shift = reg(gr.Slider(1.0, 16.0, value=12.0, step=0.5, label=t("shift")), label="shift")
            custom_steps = reg(gr.Slider(0, 12, value=0, step=1, label=t("custom_steps")), label="custom_steps")
        with gr.Row():
            refine_seed = reg(gr.Number(value=-1, precision=0, label=t("upscale_seed")), label="upscale_seed")
            anchor = reg(gr.Checkbox(value=True, label=t("anchor"), info=t("anchor.info")),
                         label="anchor", info="anchor.info")

    with gr.Row(elem_classes="aiq-actions"):
        gen_btn = reg(gr.Button(t("gen"), variant="primary", size="lg", scale=2, elem_id="gen-btn"), value="gen")
        stop_btn = reg(gr.Button(t("stop"), size="lg", scale=1, elem_id="stop-btn"), value="stop")
        clear_btn = reg(gr.Button(t("clear"), size="lg", scale=1, elem_id="clear-btn"), value="clear")

    reg(gr.HTML(f'<div class="aiq-section">{t("section.result")}</div>'),
        value=lambda l: f'<div class="aiq-section">{TEXTS[l]["section.result"]}</div>')
    with gr.Row(elem_classes="aiq-videos"):
        preview_out = reg(gr.Video(label=t("preview.label")), label="preview.label")
        final_out = reg(gr.Video(label=t("final.label")), label="final.label")
    cmp_title = reg(gr.HTML(f'<div class="aiq-section">{t("compare.title")}</div>'),
                    value=lambda l: f'<div class="aiq-section">{TEXTS[l]["compare.title"]}</div>')
    compare_out = gr.HTML(compare_idle())
    deliver_btn = gr.Button("deliver", elem_id="aiq-deliver")  # hidden; clicked by the page script
    with gr.Group(elem_classes="aiq-card"):
        with gr.Row():
            picker = reg(gr.Dropdown(list_previews(), value=(list_previews() or [(None, None)])[0][1], scale=4,
                                     label=t("picker.label"), info=t("picker.info")),
                         label="picker.label", info="picker.info")
            refresh_btn = gr.Button("🔄", scale=0, min_width=60)
        picker_info = gr.Markdown("")
        upscale_btn = reg(gr.Button(t("upscale.btn"), size="lg", elem_id="upscale-btn"), value="upscale.btn")
    latest_btn = reg(gr.Button(t("latest.btn")), value="latest.btn")
    status_out = reg(gr.Textbox(label=t("status.label"), value=t("ready"), lines=7, interactive=False,
                                elem_id="aiq-status"), label="status.label")
    seen_job = gr.State([0, 0, 0])
    poll_timer = gr.Timer(2.0)

    with gr.Accordion(t("examples"), open=False, elem_classes="aiq-acc") as acc_ex:
        reg(acc_ex, label="examples")
        gr.Examples(EXAMPLES, inputs=[prompt, dialogue, language, sound, music], label="")
    reg(gr.HTML(footer_html()), value=lambda l: footer_html())

    lora1.change(lora_info, lora1, [s1, lora1_info])
    lora2.change(lora_info, lora2, [s2, lora2_info])
    for control in (output, preset, duration, aspect):
        control.change(eta_text, [output, preset, duration, aspect], eta, queue=False, show_progress="hidden")
    for control in (picker, output, preset):
        control.change(picker_eta, [picker, output, preset], picker_info, queue=False, show_progress="hidden")
    gen_btn.click(
        start_generation,
        inputs=[prompt, dialogue, language, sound, music, start_img, end_img, aspect, duration, seed, lora1, s1, lora2,
                s2, custom_url, custom_file, s3, auto_trigger, sampler, shift, custom_steps,
                output, preset, refine_seed, anchor],
        outputs=status_out, queue=False)
    upscale_btn.click(start_upscale, inputs=[picker, output, preset, refine_seed, anchor], outputs=status_out, queue=False)
    upload_btn.click(start_upload, inputs=[upload_video, prompt, dialogue, language, sound, music, duration, seed, lora1, s1,
                                           lora2, s2, custom_url, custom_file, s3, auto_trigger, output, preset, refine_seed],
                     outputs=status_out, queue=False)
    deliver_btn.click(deliver_compare, outputs=compare_out, queue=False, show_progress="hidden")
    poll_timer.tick(poll_job, inputs=seen_job, outputs=[preview_out, final_out, status_out, seen_job, picker], queue=False,
                    show_progress="hidden")
    stop_btn.click(stop_generation, outputs=status_out, queue=False)
    refresh_btn.click(refresh_previews, outputs=picker, queue=False)
    latest_btn.click(load_latest_video, outputs=[preview_out, final_out, status_out, compare_out], queue=False)
    clear_btn.click(lambda: (None, None, t("ready"), "", "", "", "", None, None, -1, compare_idle()),
                    outputs=[preview_out, final_out, status_out, prompt, dialogue, sound, music, start_img, end_img, seed,
                             compare_out],
                    queue=False)
    lang_radio.change(switch_lang, [lang_radio, output, preset, duration, aspect, picker],
                      [c for c, _ in I18N] + [eta, picker_info, compare_out],
                      js=LANG_JS, queue=False, show_progress="hidden")
