# Step 1: Environment, pinned Wan2GP, dependencies and model downloads

import os
import sys
import time
import json
import shutil
import struct
import hashlib
import subprocess
import threading
import urllib.request
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed

ENGINE_FILE = "run_h3_2k.py"


def stop_old_engine():
    # A previous run's engine (or one left behind by a kernel restart) still holds both GPUs and ~24 GB of RAM.
    stale = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            args, name = p.info["cmdline"] or [], (p.info["name"] or "").lower()
            engine = "python" in name and any(os.path.basename(a) == ENGINE_FILE for a in args[1:3])
            if p.pid != os.getpid() and (engine or name == "cloudflared" or name.startswith("frpc")):
                p.kill()
                stale.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.wait_procs(stale, timeout=20)
    if stale:
        print(f"🧹 Stopped {len(stale)} leftover engine process(es) from an earlier run: GPU memory and RAM are free again")


stop_old_engine()

# ---- Settings ----
WAN2GP_COMMIT = "b8b18f8114e432eea8f3d7e853a51dd91fa99571"  # deepbeepmeep/Wan2GP, 1 Oct 2026 (tested)
OPEN_H3 = ("Efficient-Large-Model/Open-H3-Regenerate-2K", "5809c33730c18b6b1c827b40974855a4288641b3")  # NVIDIA, 28 Oct 2026
SHARP_TABLE = ("Comfy-Org/MiniMax-H3", "e5eb578a89295337b8ff433a035929ce0279e0b6",
               "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
               "c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7")  # Sharp's 1025 x 8 curve table
REVISIONS = {  # every Hugging Face repo is read at a fixed revision
    "DeepBeepMeep/MiniMax-H3": "5905b204b04d019c69707d89d49be4c8ae1b134f",  # 28 Oct 2026
    "pdmd2026/pdmd_4NFE_lora": "9913253b4cee1702671c38637bf3941e1de489f5",  # 4 Oct 2026
    OPEN_H3[0]: OPEN_H3[1],
}
ROOT = "/kaggle/tmp" if os.path.isdir("/kaggle") else os.path.abspath(".")
MODELS_DIR, LORA_DIR, REGEN_DIR = (os.path.join(ROOT, d) for d in ("h3_models", "h3_loras", "h3_regen"))

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"  # no second copy of 40 GB in the xet cache
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=" * 70)
print("🎬 MiniMax H3 + Open-H3 Regenerate 2K - Video + Audio")
print("📺 Created by: AiWithMase | YouTube & Telegram: @AiWithMase")
print("=" * 70)
print("🎬 Preview: MiniMax H3 base transformer + PDMD 4-step LoRA (384p, with audio)")
print("🔍 Upscale: NVIDIA Open-H3-Regenerate-2K (refiner LoRA + 3D latent upscaler) → 720p / 1080p / 2K")
print("=" * 70)
ram = psutil.virtual_memory()
print(f"RAM: {ram.total / 1024**3:.1f} GB total, {ram.available / 1024**3:.1f} GB available")
try:
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip().splitlines()
    for i, line in enumerate(smi):
        print(f"✅ GPU {i}: {line}")
    if len(smi) < 2:
        print("❌ This notebook needs 2 GPUs: Settings → Accelerator → GPU T4 x2")
except Exception:
    print("❌ No GPU found: Settings → Accelerator → GPU T4 x2")

# ---- 1. Wan2GP, pinned to a tested commit ----
t0 = time.time()
if not os.path.isdir("Wan2GP/.git"):
    print("\n📥 Fetching Wan2GP...")
    subprocess.run(["git", "init", "-q", "Wan2GP"], check=True)
    subprocess.run(["git", "-C", "Wan2GP", "remote", "add", "origin", "https://github.com/DeepBeepMeep/Wan2GP.git"], check=True)
subprocess.run(["git", "-C", "Wan2GP", "fetch", "-q", "--depth", "1", "origin", WAN2GP_COMMIT], check=True)
subprocess.run(["git", "-C", "Wan2GP", "reset", "-q", "--hard", "FETCH_HEAD"], check=True)  # also undoes stale edits on re-runs
print(f"✅ Wan2GP @ {WAN2GP_COMMIT[:10]} ({time.time() - t0:.0f}s)")

# ---- 2. Dependencies: only what the H3 path imports ----
# (Wan2GP's full requirements.txt pins an onnxruntime nightly that is no longer on PyPI.)
print("📦 Installing dependencies...")
t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts", "--timeout", "120", "--retries", "5",
                "mmgp==3.8.2", "comfy-kitchen==0.2.35", "transformers==4.54.0", "diffusers==0.36.0", "gguf==0.17.1", "optimum-quanto", "accelerate>=1.1.1", "gradio==5.29.0", "huggingface_hub[hf_xet]", "hf_xet>=1.5.2", "sentencepiece", "soundfile"], check=True)
print(f"✅ Dependencies installed ({time.time() - t0:.0f}s)")

# ---- 3. Model weights (parallel; files from an attached Kaggle Dataset are reused) ----
H3 = "DeepBeepMeep/MiniMax-H3"
TE = "Qwen3-VL-32B-Instruct"
DIT = (H3, "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors", 21.06)  # the base T2VA / FL2VA transformer
# Preview distillation LoRA: PDMD 4-step (UCSD / ByteDance: https://github.com/ZeamoxWang/pdmd), in its own folder
# because the file has a generic name.
PDMD_LORAS = [(("pdmd2026/pdmd_4NFE_lora", "lora_model_0.safetensors", 1.38), os.path.join(LORA_DIR, "pdmd_4nfe"))]
UPSCALER = (OPEN_H3[0], "latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors", 0.69)

BIG = [DIT,
       (H3, f"{TE}/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf", 14.58),
       (H3, "minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors", 2.81),
       (H3, "MiniMax-H3-audio_vae_fp32.safetensors", 0.61)]
SMALL = [(H3, f"{TE}/{name}", 0.0) for name in
         ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "preprocessor_config.json", "chat_template.json")]
JOBS = [(item, MODELS_DIR) for item in BIG + SMALL] + PDMD_LORAS + [(UPSCALER, REGEN_DIR)]
for d in (MODELS_DIR, LORA_DIR, REGEN_DIR, *(d for _, d in PDMD_LORAS)):
    os.makedirs(d, exist_ok=True)

attached = {}
if os.path.isdir("/kaggle/input"):
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            attached.setdefault(f, os.path.join(root, f))


def fetch(item, local_dir):
    from huggingface_hub import hf_hub_download
    repo, filename, size_gb = item
    dest = os.path.join(local_dir, filename)
    if os.path.exists(dest):
        return f"✓ {filename}"
    source = attached.get(os.path.basename(filename)) if filename != "lora_model_0.safetensors" else None  # ambiguous
    if source and os.path.getsize(source) > 0:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.symlink(source, dest)
        return f"✓ {filename} (from attached dataset)"
    revision = REVISIONS.get(repo)
    for attempt in range(1, 4):
        try:
            hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir, revision=revision)
            return f"✅ {filename}" + (f" ({size_gb:.1f} GB)" if size_gb else "")
        except Exception as e:
            if attempt == 3:
                raise
            print(f"  ⚠️ {filename}: {e} - retrying ({attempt}/3)")
            time.sleep(5 * attempt)


def folder_bytes(*paths):
    total = 0
    for path in paths:
        for root, _, files in os.walk(path):
            for f in files:
                p = os.path.join(root, f)
                if not os.path.islink(p):
                    try:
                        total += os.path.getsize(p)
                    except OSError:
                        pass
    return total


def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


print_lock = threading.Lock()
bar_shown = [False]


def say(message):  # a normal line printed above the live bar
    with print_lock:
        if bar_shown[0]:
            sys.stdout.write("\r" + " " * 100 + "\r")
            bar_shown[0] = False
        print(message, flush=True)


def live_bar(done, total, start, samples, unit="GB"):
    """One live line: done / total, %, smoothed speed, ETA."""
    now = time.time()
    samples[:] = [(t, b) for t, b in samples if now - t <= 10] + [(now, done)]
    speed = (samples[-1][1] - samples[0][1]) / max(samples[-1][0] - samples[0][0], 1e-6) if len(samples) > 1 else 0
    done = min(done, total)
    frac = done / max(total, 1.0)
    fill = int(frac * 28)
    eta = fmt_time(max(0.0, total - done) / speed) if speed > 1e6 else "--"
    line = (f"  ⬇️ [{'█' * fill}{'░' * (28 - fill)}] {done / 1e9:5.1f} / {total / 1e9:.1f} {unit} {frac * 100:3.0f}% | "
            f"{speed / 1e6:4.0f} MB/s | ETA {eta} | {fmt_time(now - start)}")
    with print_lock:
        sys.stdout.write("\r" + line.ljust(100))
        sys.stdout.flush()
        bar_shown[0] = True


todo = [(item, d) for item, d in JOBS
        if not os.path.exists(os.path.join(d, item[1]))
        and (item[1] == "lora_model_0.safetensors" or os.path.basename(item[1]) not in attached)]
need_gb = sum(item[2] for item, _ in todo)
try:  # exact sizes for an exact progress bar (the sizes above are rounded)
    from huggingface_hub import HfApi
    exact = 0
    for repo in {item[0] for item, _ in todo}:
        files = [item[1] for item, _ in todo if item[0] == repo]
        revision = REVISIONS.get(repo)
        exact += sum(info.size or 0 for info in HfApi().get_paths_info(repo, files, revision=revision))
    need_gb = exact / 1e9 if exact else need_gb
except Exception:
    pass
REFINE_INT8 = os.path.join(REGEN_DIR, "open_h3_regenerate_2k_lora_int8.safetensors")
SHARP_FILE = os.path.join(REGEN_DIR, "open_h3_sharp_curve_deltas.safetensors")
regen_gb = (3.2 if not os.path.exists(REFINE_INT8) else 0) + (0.2 if not os.path.exists(SHARP_FILE) else 0)
free_gb = shutil.disk_usage(ROOT).free / 1024**3
print(f"\n🚀 Downloading {need_gb:.1f} GB of weights to {ROOT} ({free_gb:.0f} GB free)...")
if need_gb + regen_gb > free_gb - 2:
    raise SystemExit(f"❌ Not enough disk space: need {need_gb + regen_gb:.0f} GB, {free_gb:.0f} GB free.")
done = threading.Event()


def monitor():
    # hf_xet writes into partial files, so bytes on disk (minus what was already there) is the real progress.
    base = folder_bytes(MODELS_DIR, LORA_DIR, REGEN_DIR)
    start, samples = time.time(), []
    while not done.wait(1.0):
        live_bar(folder_bytes(MODELS_DIR, LORA_DIR, REGEN_DIR) - base, max(need_gb * 1e9, 1.0), start, samples)


threading.Thread(target=monitor, daemon=True).start()
t0 = time.time()
try:
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(fetch, item, d) for item, d in JOBS]
        for job in jobs:
            say("  " + job.result())
finally:
    done.set()
    time.sleep(1.1)
say(f"  ✅ {need_gb:.1f} GB downloaded in {fmt_time(time.time() - t0)}"
    + (f" (avg {need_gb * 1e3 / max(time.time() - t0, 1):.0f} MB/s)" if need_gb else ""))
for d in (MODELS_DIR, LORA_DIR, REGEN_DIR):
    shutil.rmtree(os.path.join(d, ".cache"), ignore_errors=True)


# ---- 4. Open-H3-Regenerate-2K: refiner LoRA -> INT8, Sharp curve branches -> 50 small matrices ----
# The release LoRA is one 12.45 GB FP32 file (rank 576 on 312 layers). The engine keeps it on the GPUs as INT8 with
# one scale per row (0.2% error on the LoRA's delta), so it is converted while it downloads: every tensor is fetched by
# HTTP byte range, quantized and written in place, and the FP32 file never touches the disk (3.1 GB instead of 12.45).
# Sharp's 50 AdaLN branches (1.24 GB) only ever act through B @ A on an 8-D curve, so each is stored as that
# [96768 x 8] product (155 MB in all), next to the 1025 x 8 Ref2VA curve table they read their input from.
import numpy as np

OPEN_H3_FILES = {"lora": "lora/open_h3_regenerate_2k_lora.safetensors", "sharp": "lora/sharp_curve_branches.safetensors"}
LORA_SOURCE = {"bytes": 12452449080, "sha256": "8e36a0912ab72f7f3b00f41e3c7c97c7dfdec21e66c88ab22f08b9b0c681bb04"}
FORMAT = "int8-rowwise-v1"


class Source:
    """A safetensors file read by byte range: an attached local copy, or the pinned Hugging Face revision."""

    def __init__(self, repo, revision, filename):
        local = attached.get(os.path.basename(filename))
        self.path = local if local and os.path.getsize(local) > 0 else None
        self.url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
        n = struct.unpack("<Q", self.read(0, 8))[0]
        self.header = json.loads(self.read(8, n))
        self.header.pop("__metadata__", None)
        self.base = 8 + n

    def read(self, start, length):
        if self.path:
            with open(self.path, "rb") as f:
                f.seek(start)
                return f.read(length)
        for attempt in range(1, 7):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + length - 1}",
                                                                "Accept-Encoding": "identity"})
                with urllib.request.urlopen(req, timeout=300) as response:
                    if response.status != 206:
                        raise RuntimeError("the server ignored the byte-range request")
                    data = response.read()
                if len(data) != length:
                    raise RuntimeError(f"short read ({len(data)} of {length} bytes)")
                return data
            except Exception:
                if attempt == 6:
                    raise
                time.sleep(3 * attempt)

    def tensor(self, key):
        info = self.header[key]
        a, b = info["data_offsets"]
        if info["dtype"] != "F32":
            raise ValueError(f"{key}: expected FP32, got {info['dtype']}")
        dtype = np.float32
        return np.frombuffer(self.read(self.base + a, b - a), dtype).reshape(info["shape"])


def write_safetensors_layout(path, layout, metadata):
    """Header for [(key, dtype, shape, nbytes)] with 8-byte aligned tensors; returns {key: absolute file offset}."""
    header, offsets, pos = {"__metadata__": metadata}, {}, 0
    for key, dtype, shape, nbytes in layout:
        header[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [pos, pos + nbytes]}
        offsets[key] = pos
        pos += -(-nbytes // 8) * 8
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        f.truncate(8 + len(raw) + pos)
    return {key: 8 + len(raw) + off for key, off in offsets.items()}


def quantize_rows(t):
    t = t.astype(np.float32)
    scale = np.maximum(np.abs(t).max(axis=1) / 127.0, 1e-20).astype(np.float32)
    return np.clip(np.rint(t / scale[:, None]), -127, 127).astype(np.int8), scale


def file_ok(path, fmt, count):
    marker = path + ".json"
    try:
        with open(marker) as f:
            info = json.load(f)
        return info.get("format") == fmt and info.get("tensors") == count and os.path.getsize(path) == info.get("bytes")
    except Exception:
        return False


def run_parallel(keys, work, total_bytes, label):
    start, samples, got = time.time(), [], [0]
    lock = threading.Lock()

    def wrapped(key):
        n = work(key)
        with lock:
            got[0] += n
            live_bar(got[0], total_bytes, start, samples)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for job in as_completed([pool.submit(wrapped, k) for k in keys]):
            job.result()
    say(f"  ✅ {label} in {fmt_time(time.time() - start)}")


if os.path.basename(REFINE_INT8) in attached and not os.path.exists(REFINE_INT8):
    for suffix in ("", ".json"):
        if os.path.basename(REFINE_INT8) + suffix in attached:
            os.symlink(attached[os.path.basename(REFINE_INT8) + suffix], REFINE_INT8 + suffix)
if file_ok(REFINE_INT8, FORMAT, 1248):
    say("  ✓ Open-H3 refiner LoRA (INT8)")
else:
    say("  🔁 Open-H3 refiner LoRA: 12.45 GB FP32 → 3.1 GB INT8, converted while it downloads...")
    src = Source(*OPEN_H3, OPEN_H3_FILES["lora"])
    keys = sorted(src.header)
    if len(keys) != 624 or any(src.header[k]["dtype"] != "F32" for k in keys):
        raise SystemExit("❌ Unexpected Open-H3 LoRA layout (expected 624 FP32 tensors)")
    layout = []
    for k in keys:
        rows, cols = src.header[k]["shape"]
        layout += [(k, "I8", (rows, cols), rows * cols), (k + "_scale", "F32", (rows,), rows * 4)]
    tmp = REFINE_INT8 + ".part"
    offsets = write_safetensors_layout(tmp, layout, {"format": FORMAT, "source": f"{OPEN_H3[0]}@{OPEN_H3[1]}",
                                                     "source_sha256": LORA_SOURCE["sha256"]})
    fd = os.open(tmp, os.O_WRONLY)

    def convert(key):
        t = src.tensor(key)
        if not np.isfinite(t).all():
            raise ValueError(f"non-finite values in {key}")
        q, scale = quantize_rows(t)
        os.pwrite(fd, q.tobytes(), offsets[key])
        os.pwrite(fd, scale.tobytes(), offsets[key + "_scale"])
        return t.nbytes

    try:
        run_parallel(keys, convert, LORA_SOURCE["bytes"], "Open-H3 refiner LoRA converted to INT8")
    finally:
        os.close(fd)
    os.replace(tmp, REFINE_INT8)
    with open(REFINE_INT8 + ".json", "w") as f:
        json.dump({"format": FORMAT, "tensors": len(layout), "bytes": os.path.getsize(REFINE_INT8),
                   "source_sha256": LORA_SOURCE["sha256"]}, f)

if file_ok(SHARP_FILE, "sharp-deltas-v1", 51):
    say("  ✓ LMS Sharp curve branches")
else:
    say("  🔁 LMS Sharp curve branches: 1.24 GB → 50 matrices B @ A (155 MB)...")
    src = Source(*OPEN_H3, OPEN_H3_FILES["sharp"])
    roots = sorted({k[:-len(".curve_A.weight")] for k in src.header if k.endswith(".curve_A.weight")})
    if len(roots) != 50:
        raise SystemExit("❌ Unexpected Sharp curve layout (expected 50 branches)")
    deltas = {}

    def product(root):
        a, b = src.tensor(root + ".curve_A.weight"), src.tensor(root + ".curve_B.weight")
        block = int(root.split("transformer_blocks.")[1].split(".")[0])
        deltas[f"blocks.{block}.delta"] = (b.astype(np.float64) @ a.astype(np.float64)).astype(np.float32)
        return a.nbytes + b.nbytes

    run_parallel(roots, product, 1238746768, "Sharp curve branches")
    table_src = Source(SHARP_TABLE[0], SHARP_TABLE[1], SHARP_TABLE[2])
    info = table_src.header["adaln_t_table"]
    a, b = info["data_offsets"]
    raw = table_src.read(table_src.base + a, b - a)
    if info["shape"] != [1025, 8] or hashlib.sha256(raw).hexdigest() != SHARP_TABLE[3]:
        raise SystemExit("❌ The Ref2VA curve table does not match the one Open-H3 was validated with")
    deltas["ref2va_adaln_t_table"] = np.frombuffer(raw, np.float32).reshape(1025, 8).copy()
    from safetensors.numpy import save_file
    save_file(deltas, SHARP_FILE + ".part", metadata={"format": "sharp-deltas-v1"})
    os.replace(SHARP_FILE + ".part", SHARP_FILE)
    with open(SHARP_FILE + ".json", "w") as f:
        json.dump({"format": "sharp-deltas-v1", "tensors": len(deltas), "bytes": os.path.getsize(SHARP_FILE)}, f)
    del deltas

# Settings for Step 2's engine (also saved to a file, so Step 2 can be re-run alone after a kernel restart)
H3_SETTINGS = {"H3_MODELS_DIR": MODELS_DIR, "H3_LORA_DIR": LORA_DIR, "H3_REGEN_DIR": REGEN_DIR, "H3_DIT_FILE": DIT[1]}
os.environ.update(H3_SETTINGS)
with open("h3_2k_settings.json", "w") as f:
    json.dump(H3_SETTINGS, f)
print("\n" + "=" * 70)
print(f"✅ All weights ready | disk free: {shutil.disk_usage(ROOT).free / 1024**3:.0f} GB")
print("=" * 70)