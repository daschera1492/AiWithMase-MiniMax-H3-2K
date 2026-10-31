# Step 2: High-performance engine + branded Gradio UI
# Writes run_h3_2k.py, then runs it: model load (~4-5 min) → public links.
import codecs
import os
import re
import sys
import subprocess

RUN_SCRIPT_CODE = r'''"""MiniMax H3 + Open-H3-Regenerate-2K video + audio engine for Kaggle GPU T4 x2 - AiWithMase.

Step 2 of the notebook writes this file as run_h3_2k.py and runs it. It drives Wan2GP's MiniMax H3 code directly:

* Preview: the pruned 20B H3 base transformer (FL2VA / T2VA partition, INT8 ConvRot) with the PDMD 4-step
  distillation LoRA makes a 384p video with audio. Your own low-res video can be used instead: it is encoded by the
  H3 VAE and goes through the same upscale.
* Upscale (NVIDIA Open-H3-Regenerate-2K): the preview latent goes through the 3D-conv H3 latent upscaler to the
  720p / 1080p / 2K grid, is re-noised to sigma 0.909 and regenerated in 3 Euler steps by the same transformer with
  the Open-H3 refiner LoRA (refiner + LightX2V FL2VA Turbo + LMS Sharp, merged, rank 576) and Sharp's AdaLN curve
  branches. The upscaled frame is processed as overlapping spatial tiles whose predictions are blended every step,
  so attention cost stays manageable on T4s. The preview's audio is kept.
* Every INT8 linear runs on Comfy Kitchen's INT8 tensor-core GEMM (W8A8 with the ConvRot Hadamard rotation).
* Activations stay in BF16 like upstream H3. Only attention runs in FP16 (Turing has no BF16 attention kernel);
  q and k are RMS-normalized there, so FP16 is safe. The residual stream never sees FP16, so it cannot overflow.
* Tensor parallel DiT: every block is split in half across the two T4s (attention heads and FFN columns), so both GPUs
  compute every block at the same time and only exchange the block's partial outputs. Nothing is streamed per step.
* Qwen3-VL-32B (GGUF Q4_K_M) and the VAEs are pinned in RAM and run by MMGP on GPU 1, unloaded before denoising.
"""
import copy
import faulthandler
import functools
import gc
import glob
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import warnings

warnings.filterwarnings("ignore")
faulthandler.enable(file=sys.__stderr__, all_threads=True)  # a hard crash (segfault, abort) prints where it happened
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

WAN2GP_DIR = os.path.abspath("Wan2GP")
WORK_DIR = os.path.dirname(WAN2GP_DIR)
OUTPUT_DIR = os.path.join(WORK_DIR, "outputs")
MODELS_DIR = os.environ.get("H3_MODELS_DIR", "/kaggle/tmp/h3_models")
LORA_DIR = os.environ.get("H3_LORA_DIR", "/kaggle/tmp/h3_loras")
REGEN_DIR = os.environ.get("H3_REGEN_DIR", "/kaggle/tmp/h3_regen")
DIT_FILE = os.environ.get("H3_DIT_FILE", "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors")
TE_FILE = os.environ.get("H3_TE_FILE", "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf")
PDMD_FILE = os.path.join(LORA_DIR, "pdmd_4nfe", "lora_model_0.safetensors")  # PDMD 4-step distillation LoRA
REFINE_LORA_FILE = os.path.join(REGEN_DIR, "open_h3_regenerate_2k_lora_int8.safetensors")
SHARP_FILE = os.path.join(REGEN_DIR, "open_h3_sharp_curve_deltas.safetensors")
UPSCALER_FILE = os.path.join(REGEN_DIR, "latent_upsampler", "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors")
PIN_TEXT_ENCODER = os.environ.get("H3_PIN_TE", "1") == "1"
PREVIEW_DIR = os.path.join(OUTPUT_DIR, "previews")  # preview latents + prompt states, upscalable later
SERVER_PORT = 7860
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(OUTPUT_DIR, ".gradio_cache"))
if WAN2GP_DIR not in sys.path:
    sys.path.insert(0, WAN2GP_DIR)
os.chdir(WAN2GP_DIR)

import numpy as np
import psutil
import torch

import gradio as gr


def _gb(n):
    return f"{n / 1024**3:.1f} GB"


def _ram_in_use():
    """total - available: unlike psutil's 'used', this counts pinned (page-locked) memory, which Linux files as shared."""
    vm = psutil.virtual_memory()
    return f"{_gb(vm.total - vm.available)} / {_gb(vm.total)}"


def _fmt_secs(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


class ConsoleBar:
    """One live progress line per phase, redrawn every second with a carriage return (Kaggle's cell output overwrites it in place).
    A finished phase is replaced by a one-line summary. Step-based phases (denoising) interpolate inside the current
    step from the previous step's duration, so the bar keeps moving and the ETA counts down between callbacks."""

    WIDTH, LINE = 24, 104
    # phase -> seconds before a stack dump (hang detector). A 2K decode streams into the MP4 and takes several minutes.
    WATCHED = {"decode": 1800, "audio": 300, "save": 300}

    def __init__(self, stream):
        self.out = stream
        self.lock = threading.RLock()
        self.phase = None
        self.visible = False
        self.text_pending = False  # another writer is in the middle of a line: don't draw over it
        threading.Thread(target=self._loop, daemon=True).start()

    def start(self, key, label, total=None, unit=""):
        with self.lock:
            if self.phase is not None and self.phase["key"] == key:
                return
            self.finish()
            now = time.time()
            self.phase = dict(key=key, label=label, total=total, unit=unit, done=0, t0=now, t_done=now, dt=None)
            if key in self.WATCHED:  # a phase that should take seconds: dump every thread's stack if it hangs
                faulthandler.dump_traceback_later(self.WATCHED[key], file=sys.__stderr__)
            self._draw()

    def update(self, done, total=None):
        with self.lock:
            p = self.phase
            if p is None:
                return
            now = time.time()
            if total:
                p["total"] = total
            if done > p["done"]:
                p["dt"] = (now - p["t_done"]) / (done - p["done"])
                p["t_done"], p["done"] = now, done
            self._draw()

    def finish(self):
        with self.lock:
            p, self.phase = self.phase, None
            if p is None:
                return
            faulthandler.cancel_dump_traceback_later()
            elapsed = time.time() - p["t0"]
            line = f"  ✅ {p['label']}"
            if p["total"]:
                line += f" {p['done']}/{p['total']} {p['unit']}"
            line += f" in {_fmt_secs(elapsed)}"
            if p["unit"] == "steps" and p["done"]:
                line += f" ({elapsed / p['done']:.1f}s/step)"
            self._clear()
            self.out.write(line + "\n")
            self.out.flush()

    def abort(self):
        """End the live line of a phase that failed (no ✅ summary)."""
        with self.lock:
            p, self.phase = self.phase, None
            if p is None:
                return
            faulthandler.cancel_dump_traceback_later()
            self._clear()
            self.out.write(f"  ❌ {p['label']} failed after {_fmt_secs(time.time() - p['t0'])}\n")
            self.out.flush()

    def _render(self):
        p, now = self.phase, time.time()
        elapsed = now - p["t0"]
        if not p["total"]:
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(now * 4) % 10]
            return f"  {spinner} {p['label']} | {_fmt_secs(elapsed)}"
        progress = p["done"]
        if p["dt"] and p["done"] < p["total"]:
            progress += min(0.97, (now - p["t_done"]) / p["dt"])
        frac = min(1.0, progress / p["total"])
        fill = int(round(frac * self.WIDTH))
        line = (f"  ⏳ {p['label']:<16} [{'█' * fill}{'░' * (self.WIDTH - fill)}] {frac * 100:3.0f}% | "
                f"{p['done']}/{p['total']} {p['unit']}")
        if p["dt"]:
            eta = max(0.0, (p["total"] - p["done"]) * p["dt"] - (now - p["t_done"]))
            line += f" | {p['dt']:.1f}s each | ETA {_fmt_secs(eta)}"
        return line + f" | {_fmt_secs(elapsed)}"

    def _draw(self):
        if self.phase is None or self.text_pending:
            return
        self.out.write("\r" + self._render().ljust(self.LINE))
        self.out.flush()
        self.visible = True

    def _clear(self):
        if self.visible:
            self.out.write("\r" + " " * self.LINE + "\r")
            self.visible = False

    def _loop(self):
        while True:
            time.sleep(1.0)
            with self.lock:
                self._draw()


class _BarAwareStream:
    """stdout / stderr wrapper: ordinary log lines clear the live bar first and print above it."""

    def __init__(self, stream, bar):
        self._stream, self._bar = stream, bar

    def write(self, data):
        with self._bar.lock:
            if data:
                self._bar._clear()
                self._bar.text_pending = not data.endswith("\n")
            return self._stream.write(data)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


BAR = ConsoleBar(sys.stdout)
sys.stdout = _BarAwareStream(sys.stdout, BAR)
sys.stderr = _BarAwareStream(sys.stderr, BAR)


if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit("❌ This notebook needs Kaggle's 'GPU T4 x2' accelerator (Settings → Accelerator → GPU T4 x2).")

DIT_DEV = torch.device("cuda", 0)   # first half of the DiT blocks
MAIN_DEV = torch.device("cuda", 1)  # second half + text encoder + VAEs + sampler state
torch.cuda.set_device(MAIN_DEV)     # Wan2GP / MMGP use the current device for "cuda"

torch.backends.cuda.enable_flash_sdp(False)       # Turing (SM 7.5) has no flash kernel
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.benchmark = False

_props = torch.cuda.get_device_properties(0)
print("=" * 70)
print("🎬 MiniMax H3 + Open-H3 Regenerate 2K - Video + Audio | AiWithMase")
print(f"⚡ GPUs: 2 x {_props.name} ({_gb(_props.total_memory)} each, SM {_props.major}.{_props.minor}) | "
      f"RAM: {_gb(psutil.virtual_memory().total)} ({_gb(psutil.virtual_memory().available)} free)")
print("=" * 70)
sys.stdout.flush()

# Turing has no BF16 convolution kernels. Convolutions that reach BF16 (the Qwen3-VL patch embedding) run in FP32.
if _props.major < 8:
    def _conv_fp32_on_turing(original):
        def forward(self, input, weight, bias):
            if input.is_cuda and input.dtype == torch.bfloat16:
                bias32 = None if bias is None else bias.float()
                return original(self, input.float(), weight.float(), bias32).to(torch.bfloat16)
            return original(self, input, weight, bias)
        return forward
    for _conv in (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d):
        _conv._conv_forward = _conv_fp32_on_turing(_conv._conv_forward)

# ==== Wan2GP runtime: quantization handlers + INT8 kernels (what wgp.py does at import) ====
from mmgp import offload, quant_router

quant_router.unregister_handler(".fp8_quanto_bridge")
for _handler in ("shared.qtypes.int8_convrot", "shared.qtypes.gguf"):  # DiT + VAE decoder, Qwen3-VL GGUF
    quant_router.register_handler(_handler)
from shared.qtypes import gguf as _gguf_handler

quant_router.register_file_extension("gguf", _gguf_handler)

from shared.kernels import int8_backend

try:
    int8_backend.configure("auto", 1)
except Exception as e:
    print(f"⚠️ INT8 kernels unavailable, using the PyTorch path: {e}")
# Kitchen splits each INT8 GEMM into row tiles that fit a 16 MB scratch buffer, which on a T4 means 64-row GEMMs
# for the 28672-wide FFN. 128 MB gives ~700-row tiles: far better tensor-core use for ~100 MB of extra scratch.
int8_backend._SCRATCH_BYTES = 128 * 1024 * 1024
INT8_LABEL = {"kitchen": "Comfy Kitchen INT8 tensor cores", "triton": "Triton INT8 kernels"}.get(
    int8_backend._backend, "PyTorch INT8 fallback (slow)")
offload.shared_state["_attention"] = "sdpa"

from shared.utils import files_locator as fl

fl.set_checkpoints_paths([MODELS_DIR, LORA_DIR, "ckpts", "."])

import models.minimax_h3.minimax_h3_main as h3_main
from models.minimax_h3 import pipeline as h3_pipeline
from models.minimax_h3.minimax_h3_handler import family_handler
from models.minimax_h3.minimax_h3_main import VIDEO_VAE_INT8_FILE

# The pipeline's own two-phase mode is not used (the upscale runs in this engine): skip its copy of the upscaler.
h3_main._load_latent_upscaler = lambda *args, **kwargs: None

# ComfyUI-native pruned checkpoints also carry the AdaLN basis / mean they were compressed with. Wan2GP's model has
# no such tensors (it derives LoRA conversions from its own canonical maps), so drop them before loading.
_orig_strip_wrappers = h3_main._strip_wrappers


def _strip_wrappers(state_dict, *args, **kwargs):
    for key in [k for k in state_dict if k.endswith(("adaln_basis", "adaln_mean"))]:
        state_dict.pop(key)
    return _orig_strip_wrappers(state_dict, *args, **kwargs)


h3_main._strip_wrappers = _strip_wrappers

# ==== Text encoder compute dtype ====
# Without llama.cpp kernels (no wheel for Kaggle's Python), each GGUF linear is dequantized to a dense weight and run
# as a normal matmul. Turing has no BF16 tensor cores, so a BF16 GEMM can be slower than plain FP32. Measure both on
# the real layer shape and keep the faster one (FP32 is also the more precise of the two).
def _gemm_seconds(dtype, m=512, k=5120, n=25600, reps=3):
    a = torch.randn(m, k, device=MAIN_DEV, dtype=dtype)
    b = torch.randn(n, k, device=MAIN_DEV, dtype=dtype)
    torch.nn.functional.linear(a, b)
    torch.cuda.synchronize(MAIN_DEV)
    t = time.time()
    for _ in range(reps):
        torch.nn.functional.linear(a, b)
    torch.cuda.synchronize(MAIN_DEV)
    return (time.time() - t) / reps


try:
    _bf16_s, _fp32_s = _gemm_seconds(torch.bfloat16), _gemm_seconds(torch.float32)
    TE_DTYPE = torch.float32 if _fp32_s < _bf16_s else torch.bfloat16
except Exception as e:
    TE_DTYPE = torch.bfloat16
    print(f"⚠️ GEMM benchmark failed ({e}), text encoder stays in BF16")
torch.cuda.empty_cache()
_orig_load_text_encoder = h3_main._load_text_encoder
h3_main._load_text_encoder = lambda filename, dtype: _orig_load_text_encoder(filename, TE_DTYPE)

# ==== Load the model ====
# The Open-H3 refiner LoRA was trained on the base T2VA transformer, so both stages share the FL2VA partition.
MODEL_TYPE = "minimax_h3_fl2va_pruned"
DEFAULT_SAMPLER = "euler"
PREVIEW_STEPS = 4  # PDMD 4-step: Euler, video shift 12 (its sampling setting)
if not os.path.exists(PDMD_FILE):
    raise SystemExit(f"❌ The PDMD LoRA is missing ({PDMD_FILE}): re-run Step 1, then Step 2.")
print("\n📦 Loading the MiniMax H3 base transformer INT8 ConvRot + Qwen3-VL-32B GGUF...")
sys.stdout.flush()
_t_load = time.time()
BAR.start("load", "Reading checkpoints")
model_def = family_handler.query_model_def(MODEL_TYPE, {})
model_def["qkv_splitting"] = False             # keep the fused QKV GEMM: bigger tiles, and LoRAs map 1:1
model_def["video_vae_file"] = VIDEO_VAE_INT8_FILE  # INT8 ConvRot VAE decoder: half the VRAM, Kitchen GEMMs
pipeline, _ = family_handler.load_model(
    model_filename=[os.path.join(MODELS_DIR, DIT_FILE)],
    model_type=MODEL_TYPE,
    base_model_type=MODEL_TYPE,
    model_def=model_def,
    dtype=torch.bfloat16,
    VAE_dtype=torch.float32,
    text_encoder_filename=os.path.join(MODELS_DIR, "Qwen3-VL-32B-Instruct", TE_FILE),
)
transformer = pipeline.transformer
transformer.cache = None

# Modules H3 pins to FP32 (patch projections, output heads, AdaLN curve projections) are converted here,
# since this transformer is not handed to MMGP's profile (which would otherwise apply the locks).
for _module in transformer.modules():
    _lock = getattr(_module, "_lock_dtype", None)
    if _lock is not None and isinstance(_module, torch.nn.Linear) and _module.weight.dtype.is_floating_point:
        _module.to(_lock)

# ==== Tensor-parallel DiT: each block is split in half across the two T4s ====
# Within a block, GPU 0 computes attention heads 0-27 and FFN columns 0-7167, GPU 1 the other half. Both start from
# the same residual stream (kept as a replica on each GPU), and after attention and after the FFN each GPU sends its
# partial output projection to the other, so both GPUs hold the full sum. The weight slices follow the ConvRot
# layout exactly: QKV / FFN-up are split by output rows (Q, K, V are grouped), output / FFN-down by input columns on
# 256-wide rotation-group boundaries, so every INT8 GEMM computes precisely its share of the full product.
from models.minimax_h3.transformer import MLP as H3MLP, Attention as H3Attention, _gated_residual, _norm_modulate, _take
from shared.qtypes.int8_convrot import _rotate_activation

DEVICES = (DIT_DEV, MAIN_DEV)
_tp_cache = {"rope": None, "rope_copy": None, "temb": None, "temb_copy": None, "src": None, "replica": None}


class Int8Shard(torch.nn.Module):
    """One GPU's slice of an INT8 ConvRot linear (a set of output rows, or a range of input columns)."""

    def __init__(self, linear, device, rows=None, cols=None):
        super().__init__()
        data, scale = linear.weight._data, linear.weight._scale.reshape(-1)
        if rows is not None:
            data, scale = data.index_select(0, rows), scale.index_select(0, rows)
        if cols is not None:
            data = data[:, cols]
        # copy=True: own storage even on the same device (never a view into the memory-mapped checkpoint)
        self.register_buffer("qdata", data.contiguous().to(device, copy=True), persistent=False)
        self.register_buffer("qscale", scale.float().contiguous().to(device, copy=True), persistent=False)
        self.group = int(getattr(linear, "_convrot_group_size", 0) or 0)
        self.out_features, self.in_features = self.qdata.shape

    def forward(self, x):
        x2 = x.reshape(-1, self.in_features)
        if int8_backend.kitchen_enabled() and self.group == 256 and x2.is_cuda:
            out = int8_backend._linear_impl(x2, self.qdata, self.qscale, None, True)
        else:  # reference path (no Kitchen): rotate, dequantize, matmul
            weight = self.qdata.to(x2.dtype) * self.qscale.to(x2.dtype)[:, None]
            out = torch.nn.functional.linear(_rotate_activation(x2, self.group), weight)
        return out.reshape(*x.shape[:-1], self.out_features)


def _copy_to(value, dev, slot):
    """Per-forward copies of temb / rope for GPU 0 (they are the same tensor object for all 50 blocks)."""
    if value is None:
        return None
    first = value[0] if isinstance(value, tuple) else value
    if first.device == dev:
        return value
    if _tp_cache[slot] is not first:
        _tp_cache[slot] = first
        _tp_cache[slot + "_copy"] = tuple(v.to(dev) for v in value) if isinstance(value, tuple) else value.to(dev)
    return _tp_cache[slot + "_copy"]


def _all_reduce_residual(hiddens, partials, gates, segments):
    """hidden += gate * (partial_0 + partial_1) on both GPUs."""
    incoming = [partials[1].to(DEVICES[0], non_blocking=True, copy=True),
                partials[0].to(DEVICES[1], non_blocking=True, copy=True)]
    out = []
    for s, dev in enumerate(DEVICES):
        with torch.cuda.device(dev):
            branch = partials[s].add_(incoming[s])
            out.append(_gated_residual([hiddens[s]], [gates[s]], [branch], segments))
    return out


# ==== Long-video mode: the same block math on 4096-row slices ====
# A normal block builds several full-length intermediates per GPU (modulated input, fused QKV, BF16 and FP16 copies of
# Q/K/V, the output-projection input, the exchanged partial sums): roughly 75-110 KB per token. For long or 720p clips
# that does not fit next to half of the model, so in this mode only what attention needs at full length is kept (the
# residual stream and FP16 Q/K/V, ~40 KB per token). Norms, projections, RoPE, the FFN and the GPU exchange run on
# 8192-row slices (~0.4 GB of temporaries). Same arithmetic, a little more Python per block; attention dominates long clips either way.
import models.minimax_h3.transformer as _h3_transformer

LONG_VIDEO = {"on": False, "rows": 8192}


def _row_chunks(n):
    step = LONG_VIDEO["rows"]
    return [(a, min(n, a + step)) for a in range(0, n, step)]


def _slice_segments(segments, a, b):
    """(start, stop, modality row) segments of the packed sequence, clipped to rows [a, b) and shifted to 0."""
    return [(max(s0, a) - a, min(s1, b) - a, row) for s0, s1, row in segments if min(s1, b) > max(s0, a)]


def _apply_rope(query, key, rope):
    """H3's rotary embedding on [1, n, heads, dim] tensors, in place (Attention.forward's unfused path)."""
    pairs = rope.shape[-2]
    cosine, sine = rope[..., 0], rope[..., 1]
    scratch = torch.empty_like(query[..., :pairs])
    for tensor in (query, key):
        first, second = tensor[..., :pairs], tensor[..., pairs:2 * pairs]
        scratch.copy_(first)
        first.mul_(cosine).addcmul_(second, sine, value=-1)
        second.mul_(cosine).addcmul_(scratch, sine)


def _sliced_attention(shard, hidden, mod, segments, rope):
    attn = shard.attn
    rows, heads, dim = hidden.shape[0], attn.heads, attn.head_dim
    inner = heads * dim
    if isinstance(rope, tuple):
        rope = rope[0]
    q, k, v = (torch.empty(1, rows, heads, dim, device=hidden.device, dtype=torch.float16) for _ in range(3))
    for a, b in _row_chunks(rows):
        h = _norm_modulate(shard.norm1, hidden[a:b], [mod[0], mod[1]], _slice_segments(segments, a, b))
        qkv = attn.qkv_proj(h)
        del h
        query, key, value = (t.view(1, b - a, heads, dim) for t in qkv.split(inner, dim=-1))
        query, key = attn.q_norm(query), attn.k_norm(key)
        if rope is not None:
            _apply_rope(query, key, rope[:, a:b])
        q[:, a:b].copy_(query)
        k[:, a:b].copy_(key)
        v[:, a:b].copy_(value)
        del qkv, query, key, value
    qkv_list = [q, k, v]
    del q, k, v
    out = _h3_transformer.pay_attention(qkv_list, recycle_q=True)[0]  # [rows, heads, dim] FP16
    partial = torch.empty_like(hidden)
    for a, b in _row_chunks(rows):
        partial[a:b] = attn.out_proj(out[a:b].reshape(b - a, inner).to(hidden.dtype))
    return partial


def _sliced_mlp(shard, hidden, mod, segments):
    partial = torch.empty_like(hidden)
    for a, b in _row_chunks(hidden.shape[0]):
        h = _norm_modulate(shard.norm2, hidden[a:b], [mod[3], mod[4]], _slice_segments(segments, a, b))
        partial[a:b] = shard.mlp([h])
    return partial


def _all_reduce_sliced(hiddens, partials, gates, segments):
    """_all_reduce_residual without full-length copies: the partial sums cross between GPUs slice by slice."""
    for a, b in _row_chunks(hiddens[0].shape[0]):
        incoming = [partials[1][a:b].to(DEVICES[0], non_blocking=True, copy=True),
                    partials[0][a:b].to(DEVICES[1], non_blocking=True, copy=True)]
        for s, dev in enumerate(DEVICES):
            with torch.cuda.device(dev):
                partials[s][a:b].add_(incoming[s])
        del incoming
    out = []
    for s, dev in enumerate(DEVICES):
        with torch.cuda.device(dev):
            out.append(_gated_residual([hiddens[s]], [gates[s]], [partials[s]], segments))
    return out


class TPBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        attn, mlp = block.attn, block.mlp
        heads, head_dim = attn.heads, attn.head_dim
        inner, ffn, hidden = heads * head_dim, mlp.ffn, mlp.hidden
        half_inner, half_ffn = inner // 2, ffn // 2
        self.splits = []
        shards = []
        for s, dev in enumerate(DEVICES):
            h = torch.arange(s * half_inner, (s + 1) * half_inner)
            f = torch.arange(s * half_ffn, (s + 1) * half_ffn)
            qkv_rows = torch.cat((h, h + inner, h + 2 * inner))
            fc1_rows = torch.cat((f, f + ffn))
            out_cols = slice(s * half_inner, (s + 1) * half_inner)
            fc2_cols = slice(s * half_ffn, (s + 1) * half_ffn)
            self.splits.append({"qkv": qkv_rows, "fc1": fc1_rows, "out": out_cols, "fc2": fc2_cols})
            with torch.device("meta"):
                sattn = H3Attention(hidden, heads // 2, head_dim, attn.q_norm.eps, dtype=torch.bfloat16)
                smlp = H3MLP(hidden, half_ffn, mlp.chunk_size, dtype=torch.bfloat16)
            sattn.qkv_proj = Int8Shard(attn.qkv_proj, dev, rows=qkv_rows)
            sattn.out_proj = Int8Shard(attn.out_proj, dev, cols=out_cols)
            sattn.q_norm = copy.deepcopy(attn.q_norm).to(dev)
            sattn.k_norm = copy.deepcopy(attn.k_norm).to(dev)
            smlp.fc1 = Int8Shard(mlp.fc1, dev, rows=fc1_rows)
            smlp.fc2 = Int8Shard(mlp.fc2, dev, cols=fc2_cols)
            shard = torch.nn.Module()
            shard.attn, shard.mlp = sattn, smlp
            shard.norm1 = copy.deepcopy(block.norm1).to(dev)
            shard.norm2 = copy.deepcopy(block.norm2).to(dev)
            shard.adaln_proj = copy.deepcopy(block.adaln_proj).to(dev)
            shards.append(shard)
        self.shards = torch.nn.ModuleList(shards)
        self.dims = {"qkv": (hidden, 3 * inner), "out": (inner, hidden), "fc1": (hidden, 2 * ffn), "fc2": (ffn, hidden)}

    @property
    def attn(self):  # H3's forward checks block.attn for q_proj / vdn
        return self.shards[1].attn

    def forward(self, x_list, temb, segments, rope, residual_signature_elements=0):
        if residual_signature_elements:
            raise RuntimeError("First Block Cache is not supported by the tensor-parallel DiT")
        hidden = _take(x_list)
        hidden = hidden if hidden.device == MAIN_DEV else hidden.to(MAIN_DEV)
        replica = _tp_cache["replica"] if _tp_cache["src"] is hidden else hidden.to(DIT_DEV, copy=True)
        hiddens = [replica, hidden]
        tembs = [_copy_to(temb, DIT_DEV, "temb"), temb]
        ropes = [_copy_to(rope, DIT_DEV, "rope"), rope]
        if LONG_VIDEO["on"]:
            return self._forward_sliced(hiddens, tembs, ropes, segments)
        mods, partials = [], []
        for s, dev in enumerate(DEVICES):  # kernels are queued on both GPUs before either is waited for
            shard = self.shards[s]
            with torch.cuda.device(dev):
                mod = shard.adaln_proj(tembs[s])
                mods.append(mod)
                h = _norm_modulate(shard.norm1, hiddens[s], [mod[0], mod[1]], segments)
                partials.append(shard.attn([h], rope=ropes[s]))
        hiddens = _all_reduce_residual(hiddens, partials, [m[2] for m in mods], segments)
        partials = []
        for s, dev in enumerate(DEVICES):
            shard = self.shards[s]
            with torch.cuda.device(dev):
                h = _norm_modulate(shard.norm2, hiddens[s], [mods[s][3], mods[s][4]], segments)
                partials.append(shard.mlp([h]))
        hiddens = _all_reduce_residual(hiddens, partials, [m[5] for m in mods], segments)
        _tp_cache["src"], _tp_cache["replica"] = hiddens[1], hiddens[0]
        return hiddens[1]

    def _forward_sliced(self, hiddens, tembs, ropes, segments):
        mods, partials = [], []
        for s, dev in enumerate(DEVICES):  # kernels are queued on both GPUs before either is waited for
            with torch.cuda.device(dev):
                mods.append(self.shards[s].adaln_proj(tembs[s]))
                partials.append(_sliced_attention(self.shards[s], hiddens[s], mods[s], segments, ropes[s]))
        hiddens = _all_reduce_sliced(hiddens, partials, [m[2] for m in mods], segments)
        partials = []
        for s, dev in enumerate(DEVICES):
            with torch.cuda.device(dev):
                partials.append(_sliced_mlp(self.shards[s], hiddens[s], mods[s], segments))
        hiddens = _all_reduce_sliced(hiddens, partials, [m[5] for m in mods], segments)
        _tp_cache["src"], _tp_cache["replica"] = hiddens[1], hiddens[0]
        return hiddens[1]


NUM_BLOCKS = len(transformer.blocks)
BAR.start("split", "Splitting DiT", NUM_BLOCKS, "blocks")
for _i in range(NUM_BLOCKS):
    transformer.blocks[_i] = TPBlock(transformer.blocks[_i])
    BAR.update(_i + 1)
BAR.finish()
for _name, _child in transformer.named_children():
    if _name not in ("blocks", "token_refiner"):
        _child.to(MAIN_DEV)
for _name, _buffer in list(transformer.named_buffers(recurse=False)):
    transformer._buffers[_name] = _buffer.to(MAIN_DEV)
gc.collect()

# ==== Exact-size pinned RAM ====
# PyTorch's pinned-memory allocator rounds every block up to the next power of two: MMGP's ~210 MB text-encoder
# chunks become 256 MB each (+2.8 GB) and the token refiner's 17 tensors would take 2.25 GB for 1.44 GB. Page-locking
# ordinary RAM with cudaHostRegister pins exactly the bytes asked for (same fast async copies to the GPU).
_HOST_REGISTERED = []  # registered RAM stays alive (and pinned) for the whole session
PINNED_BYTES = {}


def _dont_fork(ptr, nbytes):
    """madvise(MADV_DONTFORK): a fork() must not duplicate pinned RAM. Linux cannot share pinned pages copy-on-write,
    so it copies them into the child right away: forking this process (subprocess with preexec_fn does) would need
    another ~13 GB at once and gets the engine killed. CUDA marks its own cudaHostAlloc blocks this way."""
    try:
        import ctypes
        page = os.sysconf("SC_PAGE_SIZE")
        start = -(-ptr // page) * page
        end = (ptr + nbytes) // page * page
        if end > start:
            ctypes.CDLL("libc.so.6").madvise(ctypes.c_void_p(start), ctypes.c_size_t(end - start), 10)  # MADV_DONTFORK
    except Exception:
        pass


def pinned_empty_exact(nbytes):
    t = torch.empty(nbytes, dtype=torch.uint8)
    torch.cuda.check_error(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nbytes, 0))
    _dont_fork(t.data_ptr(), nbytes)
    _HOST_REGISTERED.append(t)
    return t


_torch_empty = torch.empty


def _empty_exact_pinned(*size, pin_memory=False, **kwargs):
    """torch.empty for MMGP's pinning: its non-power-of-two uint8 chunks are registered at their exact size."""
    n = size[0] if len(size) == 1 and isinstance(size[0], int) else 0
    if pin_memory and n > 0 and n & (n - 1) and kwargs.get("dtype") == torch.uint8:
        try:
            return pinned_empty_exact(n)
        except Exception:
            pass
    return _torch_empty(*size, pin_memory=pin_memory, **kwargs)


# The 1.4 GiB token refiner runs once per prompt: keep it in pinned RAM and swap it onto GPU 1 only for that.
_refiner = transformer.token_refiner
_ref_list = list(_refiner.parameters())
_ref_offsets, _ref_total = [], 0
for _p in _ref_list:
    _ref_offsets.append(_ref_total)
    _ref_total += -(-_p.numel() * _p.element_size() // 64) * 64
try:
    _ref_buf = pinned_empty_exact(_ref_total)
    _refiner_params = []
    for _p, _off in zip(_ref_list, _ref_offsets):
        _cpu = _ref_buf[_off:_off + _p.numel() * _p.element_size()].view(_p.dtype).view(_p.shape)
        _cpu.copy_(_p.data)
        _refiner_params.append((_p, _cpu))
except Exception as e:  # no cudaHostRegister: PyTorch's (rounded) pinned blocks
    print(f"⚠️ Exact pinning unavailable ({e}); using PyTorch pinned memory")
    _refiner_params = [(p, p.data.to("cpu").pin_memory()) for p in _ref_list]
PINNED_BYTES["token refiner"] = _ref_total
for _p, _cpu in _refiner_params:
    _p.data = _cpu
_orig_preprocess_text = transformer.preprocess_text_embeds


def _preprocess_text_embeds(text_states):
    if text_states.shape[-1] == transformer.hidden_size:
        return text_states
    for p, cpu in _refiner_params:
        p.data = cpu.to(MAIN_DEV, non_blocking=True)
    try:
        return _orig_preprocess_text(text_states.to(MAIN_DEV))
    finally:
        for p, cpu in _refiner_params:
            p.data = cpu


transformer.preprocess_text_embeds = _preprocess_text_embeds

# ==== Text encoder + VAEs: MMGP on GPU 1, unloaded before denoising ====
pipeline.audio_vae._convertWeightsFloatTo = None  # keep the audio VAE in FP32 (Turing has no BF16 convs)
aux_pipe = {
    "text_encoder": pipeline.text_encoder.language_model,
    "vision_encoder": pipeline.text_encoder.visual,
    "vae": pipeline.video_decoder,
    "video_encoder": pipeline.video_encoder,
    "audio_vae": pipeline.audio_vae,
}
pipeline.video_decoder._budget = 0  # decode with the whole VAE resident (it is called once per clip and tile)
if TE_DTYPE == torch.float32:
    pipeline.text_encoder.language_model._convertWeightsFloatTo = None
    pipeline.text_encoder.visual._convertWeightsFloatTo = None
# The text encoder (13.6 GB) and the video VAE decoder (2.3 GB) are pinned in RAM. Kaggle's disk reads at ~180 MB/s,
# so a memory-mapped copy that has dropped out of the page cache costs ~15 s per video before decoding can even
# start; from pinned RAM the same weights reach the GPU in well under a second. The small audio VAE and image
# encoder stay as they are (a few seconds at most).
_pin = (["text_encoder"] if PIN_TEXT_ENCODER else []) + ["vae"]
BAR.start("pin", "Pinning text encoder + video decoder in RAM (one-time)")
try:
    torch.empty = _empty_exact_pinned
    try:
        aux_offload = offload.profile(aux_pipe, profile_no=4, quantizeTransformer=False, convertWeightsFloatTo=torch.bfloat16,
                                      pinnedMemory=_pin, asyncTransfers=True, verboseLevel=-1,
                                      budgets={"text_encoder": 1500, "vision_encoder": 0, "*": 3000})
    finally:
        torch.empty = _torch_empty
except Exception as e:
    if not _pin:
        raise
    print(f"⚠️ Pinning failed ({e}); text encoder streams from regular RAM")
    aux_offload = offload.profile(aux_pipe, profile_no=4, quantizeTransformer=False, convertWeightsFloatTo=torch.bfloat16,
                                  pinnedMemory=False, asyncTransfers=True, verboseLevel=-1,
                                  budgets={"text_encoder": 1500, "vision_encoder": 0, "*": 3000})
pipeline.set_offload_handoff(aux_offload, None)  # pipeline unloads these before handing the GPU to the DiT
BAR.finish()
for _key, _label in (("text_encoder", "text encoder"), ("vae", "video decoder")):
    PINNED_BYTES[_label] = getattr(aux_pipe[_key], "_pinned_bytes", 0)
if hasattr(torch._C, "_host_emptyCache"):
    torch._C._host_emptyCache()  # MMGP's temporary probe block

gc.collect()
for _dev in (DIT_DEV, MAIN_DEV):
    with torch.cuda.device(_dev):
        torch.cuda.empty_cache()
_free0, _free1 = torch.cuda.mem_get_info(0)[0], torch.cuda.mem_get_info(1)[0]
print(f"✅ Models loaded in {time.time() - _t_load:.0f}s | {INT8_LABEL} | DiT split over both GPUs "
      f"({_gb(_free0)} / {_gb(_free1)} free)")
sys.stdout.flush()

# ==== LoRA runtime ====
# Applied as forward hooks on the resident (two-GPU) transformer: out += strength * (x @ A^T) @ B^T, written in place
# with addmm_ so no output-sized temporary is created. Keys go through Wan2GP's own H3 converter, which maps
# diffusers / kohya names and re-projects AdaLN adapters onto the pruned 8-wide AdaLN curves.
LORA_CATALOG = {
    "🧑 Realism People (fal)": {
        "repo": "fal/MiniMax-H3-Realism-People-LoRA", "file": "h3-realism-people-t2v-i2v-r2v.safetensors",
        "scale": 1.0, "trigger": "r34l1sm", "info": "Natural skin, faces, expressions and film lighting for people shots."},
    "🎥 Camera Motion": {
        "repo": "Jojocodex/minimax-h3-Camera-Motion-lora", "file": "camera_motion_h3_lora_v1_3000_pruned.safetensors",
        "scale": 0.9, "trigger": "camera motion", "info": "Push-in, pull-back, orbit, aerial, handheld... describe the move after the trigger."},
    "🔄 360° Orbit (same image as first + last frame)": {
        "repo": "pablodawson/MiniMax-H3-360-Orbit-LoRA", "file": "minimax_h3_flf2v_lora_v1.safetensors",
        "scale": 1.0, "trigger": "One frozen instant. Only the camera moves. In a continuous 360 orbit.",
        "info": "Put the SAME image as start and end frame: the camera orbits the frozen scene and lands back on it."},
    "⚔️ Weapon Combat (BUNNY)": {
        "repo": "JOKER141/MiniMax-H3-Weapon-Combat-LoRA", "file": "Bunny_weapon_combatV1.safetensors",
        "scale": 1.0, "trigger": "BUNNY", "info": "Weapon swings, blocks, parries and dynamic fight choreography."},
    "🧱 Spatial & Physics": {
        "repo": "Jojocodex/minimax-h3-spatial-physics-lora", "file": "wushu_spatial_physics_clean_3000_pruned.safetensors",
        "scale": 0.4, "trigger": "", "info": "More physical collisions, bounces and gravity. Keep it at 0.3-0.5."},
    "🗣️ Natural Face & Speech": {
        "repo": "vpakarinen/natural-face-speech-h3-lora", "file": "natural_face_speech_h3_lora_v2_500.safetensors",
        "scale": 0.7, "trigger": "", "info": "Talking heads: facial muscle motion and clear speech. Put the line in 'Spoken line'."},
    "🎞️ Studio 1939 Animation": {
        "repo": "lovis93/studio-1939-old-animation-lora-minimax-h3", "file": "studio1939-light.safetensors",
        "scale": 1.0, "trigger": "gulliv3r,", "info": "Hand-painted golden-age animation look."},
    "🌸 Anime Motion": {
        "repo": "prithivMLmods/MiniMax-H3-I2V-Anime-Motion-LoRA", "file": "MiniMax-H3-I2V-Anime-Motion-LoRA-1400.safetensors",
        "scale": 1.0, "trigger": "Anime-Motion", "info": "Subtle looping anime motion, best with a start image."},
    "📼 VHS Tape (1980s)": {
        "repo": "KennethFal/vh5tape-vhs-lora-minimax-h3", "file": "vh5tape.safetensors",
        "scale": 1.0, "trigger": "vh5tape, worn VHS tape recording with visible tape damage.",
        "info": "Worn VHS look and sound. Damage tiers: lightly worn / worn / badly damaged."},
}
NO_LORA = "None"
_LORA_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "base_model.model.", "transformer.", "model.")
_LORA_SUFFIXES = {".default.weight": ".weight", ".lora.A.weight": ".lora_A.weight", ".lora.B.weight": ".lora_B.weight",
                  ".lora.down.weight": ".lora_down.weight", ".lora.up.weight": ".lora_up.weight"}


LORA_ROWS = 8192


class Q8:
    """A LoRA factor stored as INT8 with one FP32 scale per row (absmax / 127): half the VRAM of FP16. Measured on the
    Open-H3 refiner LoRA (real tensors): 0.2% relative error on the LoRA's weight delta, an order of magnitude below
    the error of the INT8 base weights it corrects. Rows of a row-split shard keep their scale exactly."""

    __slots__ = ("q", "s")

    def __init__(self, t):
        t = t.float()
        s = t.abs().amax(dim=1, keepdim=True).div_(127.0).clamp_min_(1e-20)
        self.q = torch.round(t / s).clamp_(-127, 127).to(torch.int8)
        self.s = s

    def half(self):
        return (self.q * self.s).to(torch.float16)  # natural magnitude, so no FP16 subnormal scales

    @property
    def shape(self):
        return self.q.shape

    def nbytes(self):
        return self.q.numel() + self.s.numel() * 4


def _f16(t):
    return t.half() if isinstance(t, Q8) else t


def _nbytes(t):
    return t.nbytes() if isinstance(t, Q8) else t.numel() * t.element_size()


class LoraRuntime:
    """LoRAs as forward hooks. Each original layer name maps to one or more target modules (the two halves of a
    tensor-parallel layer), and every target gets the matching slice of the LoRA: B rows for row-split layers, A columns
    for column-split layers (whose partial outputs are summed across GPUs, so a bias delta goes to one half only).
    The segments of one target share its input, so their A factors are concatenated into one GEMM (q / k / v)."""

    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type
        self.targets = {}  # name -> (in_features, out_features, [(module, device, dtype, kind, index)])
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and not name.startswith("blocks."):
                dev = MAIN_DEV if name.startswith("token_refiner") or not module.weight.is_cuda else module.weight.device
                dtype = torch.float32 if module.weight.dtype == torch.float32 else torch.bfloat16
                self.targets[name] = (module.in_features, module.out_features, [(module, dev, dtype, None, None)])
        for i, block in enumerate(model.blocks):
            for short, attr, kind in (("qkv", "attn.qkv_proj", "rows"), ("out", "attn.out_proj", "cols"),
                                      ("fc1", "mlp.fc1", "rows"), ("fc2", "mlp.fc2", "cols")):
                parts = [(functools.reduce(getattr, attr.split("."), block.shards[s]), DEVICES[s], torch.bfloat16, kind,
                          block.splits[s][short]) for s in range(2)]
                self.targets[f"blocks.{i}.{attr}"] = (*block.dims[short], parts)
            adaln = [(block.shards[s].adaln_proj.linear, DEVICES[s], torch.float32, None, None) for s in range(2)]
            first = adaln[0][0]
            self.targets[f"blocks.{i}.adaln_proj.linear"] = (first.in_features, first.out_features, adaln)
        self.q8_keys = set()  # LoRAs kept as INT8 factors (the large permanent ones)
        self.loaded = {}    # key -> [(module, (A, [(B, out_start, out_end, rank_start, rank_end)]), alpha_scale, diff_b)]
        self.strength = {}  # key -> current strength
        self.active = {}    # id(module) -> [(key, group, alpha_scale, diff_b)]
        self.handles = {}   # id(module) -> hook handle

    def _normalize(self, state_dict):
        out = {}
        for key, value in state_dict.items():
            for prefix in _LORA_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            for src, dst in _LORA_SUFFIXES.items():
                if key.endswith(src):
                    key = key[:-len(src)] + dst
                    break
            out[key] = value
        return self.model.preprocess_loras(self.model_type, out)

    def _iter_groups(self, path):
        """(layer name, {"A", "B", "alpha", "diff_b"}) for every layer of a LoRA file, read one block at a time: a
        large LoRA does not fit in RAM next to the pinned text encoder once Wan2GP's converter has copied it. INT8
        factors (`<key>` int8 + `<key>_scale` per row) are expanded to FP16 on the way in."""
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            batches = {}
            for k in f.keys():
                m = re.search(r"blocks[._](\d+)[._]", k)
                batches.setdefault(("refiner" if "refiner" in k else "block", int(m.group(1))) if m else ("other", 0), []).append(k)
            for keys in batches.values():
                raw = {k: f.get_tensor(k) for k in keys}
                for k in [k for k in raw if k.endswith("_scale") and k[:-len("_scale")] in raw]:
                    scale = raw.pop(k)
                    base = k[:-len("_scale")]
                    raw[base] = (raw[base] * scale.float().view(-1, 1)).to(torch.float16)
                groups = {}
                for k, v in self._normalize(raw).items():
                    for down, up in (("lora_A.weight", "lora_B.weight"), ("lora_down.weight", "lora_up.weight")):
                        if k.endswith("." + down):
                            groups.setdefault(k[:-len(down) - 1], {})["A"] = v
                        elif k.endswith("." + up):
                            groups.setdefault(k[:-len(up) - 1], {})["B"] = v
                    if k.endswith(".alpha"):
                        groups.setdefault(k[:-len(".alpha")], {})["alpha"] = float(v.item() if torch.is_tensor(v) else v)
                    elif k.endswith(".diff_b"):
                        groups.setdefault(k[:-len(".diff_b")], {})["diff_b"] = v
                del raw
                yield from groups.items()

    def _read_groups(self, path):
        return dict(self._iter_groups(path))

    @staticmethod
    def _segments(A, B):
        """[(A_i, B_i, out_start, out_end)]. Separate q / k / v adapters arrive fused as cat(A) and block_diag(B),
        which is 2/3 zeros: they are split back into their three blocks (a third of the memory and the work)."""
        r3, o3 = A.shape[0], B.shape[0]
        if r3 % 3 == 0 and o3 % 3 == 0 and r3 >= 3:
            r, o = r3 // 3, o3 // 3
            off = [B[i * o:(i + 1) * o, j * r:(j + 1) * r] for i in range(3) for j in range(3) if i != j]
            if all(not bool(t.any()) for t in off):
                return [(A[i * r:(i + 1) * r], B[i * o:(i + 1) * o, i * r:(i + 1) * r], i * o, (i + 1) * o) for i in range(3)]
        return [(A, B, 0, o3)]

    def load(self, key, path):
        q8 = key in self.q8_keys
        adapters, layers, skipped = [], 0, 0
        for name, g in self._iter_groups(path):
            target = self.targets.get(name)
            A, B, diff_b = g.get("A"), g.get("B"), g.get("diff_b")
            if target is None or (A is None) != (B is None) or (A is None and diff_b is None):
                skipped += 1
                continue
            in_f, out_f, parts = target
            if A is not None and (A.shape[1] != in_f or B.shape[0] != out_f or A.shape[0] != B.shape[1]):
                skipped += 1
                continue
            alpha_scale = g["alpha"] / A.shape[0] if A is not None and "alpha" in g else 1.0
            segments = self._segments(A, B) if A is not None else []
            for n, (module, dev, dtype, kind, index) in enumerate(parts):
                segs, d = [], diff_b
                if dtype == torch.bfloat16:  # BF16 layers get their LoRA math in FP16 on the tensor cores (see _hook)
                    dtype = torch.float16
                for a, b, o0, o1 in segments:
                    if kind == "rows":  # this shard computes the output rows `index` (in order)
                        pos = ((index >= o0) & (index < o1)).nonzero().flatten()
                        if not len(pos):
                            continue
                        assert int(pos[-1] - pos[0]) + 1 == len(pos), "non-contiguous LoRA rows in a shard"
                        b, o0, o1 = b.index_select(0, index[pos] - o0), int(pos[0]), int(pos[-1]) + 1
                    elif kind == "cols":
                        a = a[:, index]
                    segs.append((a, b, o0, o1))
                group = None
                if segs:
                    pack = (lambda t: Q8(t.to(dev))) if q8 and dtype == torch.float16 else (lambda t: t.to(dev, dtype).contiguous())
                    a_cat = pack(torch.cat([a for a, _, _, _ in segs]) if len(segs) > 1 else segs[0][0])
                    outs, r0 = [], 0
                    for a, b, o0, o1 in segs:
                        outs.append((pack(b), o0, o1, r0, r0 + a.shape[0]))
                        r0 += a.shape[0]
                    group = (a_cat, outs)
                if kind == "rows" and diff_b is not None:
                    d = diff_b.index_select(0, index)
                elif kind == "cols":
                    d = diff_b if n == 0 else None
                adapters.append((module, group, alpha_scale, None if d is None else d.to(dev, dtype)))
            layers += 1
            del A, B, segments, g
        if not adapters:
            raise ValueError(f"No layer of this LoRA matches MiniMax H3 ({skipped} groups checked)")
        self.loaded[key] = adapters
        per_dev = {}
        for module, group, _, _ in adapters:
            if group is not None:
                dev = group[0].q.device if isinstance(group[0], Q8) else group[0].device
                per_dev[dev] = per_dev.get(dev, 0) + _nbytes(group[0]) + sum(_nbytes(b) for b, *_ in group[1])
        mb = sum(per_dev.values()) / len(per_dev or [0]) / 2**20 if per_dev else 0
        print(f"  🎨 LoRA '{key}': {layers} layers ({mb:.0f} MB per GPU)"
              f"{f', {skipped} unmatched groups skipped' if skipped else ''}")
        gc.collect()

    def _hook(self, module_id):
        def hook(module, args, output):
            adapters = self.active.get(module_id)
            if not adapters:
                return output
            x = args[0]
            x2 = x.reshape(-1, x.shape[-1])
            out2 = output.view(-1, output.shape[-1])
            for key, group, alpha_scale, diff_b in adapters:
                s = self.strength.get(key, 0.0) * alpha_scale
                if s == 0.0:
                    continue
                if group is not None and (isinstance(group[0], Q8) or group[0].dtype == torch.float16):
                    # FP16 tensor cores: a rank-128 LoRA on every layer is ~3% of the FLOPs, but in FP32 (no tensor
                    # cores on Turing, and BF16 is slower still) it cost ~30% of each step. The input is scaled by a
                    # power of two before the FP16 cast and the result scaled back, so large activations (FFN-down
                    # input) cannot overflow; FP16's 11-bit mantissa is plenty for a correction term.
                    pre = 1.0 / 512 if x2.shape[-1] > 8192 else 1.0 / 64
                    a16 = _f16(group[0])
                    outs = [(_f16(b), o0, o1, r0, r1) for b, o0, o1, r0, r1 in group[1]]
                    for start in range(0, x2.shape[0], LORA_ROWS):
                        end = min(x2.shape[0], start + LORA_ROWS)
                        h = (x2[start:end] * pre).to(torch.float16) @ a16.t()
                        for b16, o0, o1, r0, r1 in outs:
                            delta = h[:, r0:r1] @ b16.t()
                            out2[start:end, o0:o1].add_(delta.to(out2.dtype), alpha=s / pre)
                            del delta
                        del h
                    del a16, outs
                elif group is not None:  # FP32 layers (AdaLN curves)
                    a32 = group[0].float()
                    for start in range(0, x2.shape[0], LORA_ROWS):
                        end = min(x2.shape[0], start + LORA_ROWS)
                        h = x2[start:end].float() @ a32.t()
                        for b32, o0, o1, r0, r1 in group[1]:
                            out2[start:end, o0:o1].add_((h[:, r0:r1] @ b32.float().t()).mul_(s).to(out2.dtype))
                        del h
                if diff_b is not None:
                    out2.add_(diff_b, alpha=s)
            return output
        return hook

    def apply(self, selections):
        """selections: {key: (path, strength)} - loads new LoRAs, frees unused ones, sets strengths."""
        for key in [k for k in self.loaded if k not in selections]:
            del self.loaded[key]
        for key, (path, strength) in selections.items():
            if key not in self.loaded:
                self.load(key, path)
            self.strength[key] = float(strength)
        self.active, modules = {}, {}
        for key, adapters in self.loaded.items():
            for module, group, alpha_scale, diff_b in adapters:
                self.active.setdefault(id(module), []).append((key, group, alpha_scale, diff_b))
                modules[id(module)] = module
        for module_id in list(self.handles):
            if module_id not in self.active:
                self.handles.pop(module_id).remove()
        for module_id, module in modules.items():
            if module_id not in self.handles:
                self.handles[module_id] = module.register_forward_hook(self._hook(module_id))
        gc.collect()
        for dev in DEVICES:
            with torch.cuda.device(dev):
                torch.cuda.empty_cache()


lora_runtime = LoraRuntime(transformer, MODEL_TYPE)

# ==== DiT attention backend: PyTorch SDPA or tensor-core GEMM attention, whichever is faster on this GPU ====
# Attention is most of the DiT's work (it grows with the square of the video length: a 10 s clip costs ~3.5x a 5 s
# one). On Turing, PyTorch's memory-efficient SDPA kernel runs well below what cuBLAS FP16 GEMMs reach, so an exact
# chunked attention (Q·Kᵀ GEMM -> softmax -> P·V GEMM, a few hundred query rows at a time) can be faster. Both are
# timed on the real head shape at startup; GEMM is used only if it is faster and matches SDPA. The VAE keeps
# Wan2GP's own attention and 256 px tiles untouched.
import shared.attention as _w2g_attention

_orig_pay_attention = _w2g_attention.pay_attention
ATTENTION = {"dit": "sdpa"}
ATTENTION_CHUNK_BYTES = 640 * 2**20  # logits + probabilities of one chunk


def gemm_attention(q, k, v):
    """Exact softmax attention for [B, L, H, D] FP16 tensors using batched GEMMs over query-row chunks."""
    batch, length, heads, dim = q.shape
    keys = k.shape[1]
    rows = max(64, min(length, ATTENTION_CHUNK_BYTES // (4 * heads * keys)))
    out = torch.empty_like(q)
    q = q * (dim ** -0.5)
    for b in range(batch):
        q_b, k_t, v_b = q[b].transpose(0, 1), k[b].permute(1, 2, 0), v[b].transpose(0, 1)  # strided views, no copies
        for start in range(0, length, rows):
            probs = torch.softmax(torch.matmul(q_b[:, start:start + rows], k_t), dim=-1)
            out[b, start:start + rows] = torch.matmul(probs, v_b).transpose(0, 1)
            del probs
    return out


def _fast_pay_attention(kind):
    def pay_attention(qkv_list, *args, **kwargs):
        if (ATTENTION[kind] != "gemm" or args or kwargs.get("attention_mask") is not None or kwargs.get("causal")
                or kwargs.get("q_lens") is not None or kwargs.get("k_lens") is not None):
            return _orig_pay_attention(qkv_list, *args, **kwargs)
        q, k, v = qkv_list
        qkv_list.clear()
        out_dtype = q.dtype
        q, k, v = [t if t.dtype == torch.float16 else t.to(torch.float16) for t in (q, k, v)]
        return gemm_attention(q, k, v).to(out_dtype)
    return pay_attention


_h3_transformer.pay_attention = _fast_pay_attention("dit")


def _time_it(fn, reps=2):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps


def benchmark_attention(kind, batch, length, heads, dim):
    try:
        with torch.cuda.device(DIT_DEV):
            q, k, v = (torch.randn(batch, length, heads, dim, device=DIT_DEV, dtype=torch.float16) for _ in range(3))
            sdpa = lambda: _orig_pay_attention([q, k, v], force_attention="sdpa")
            gemm = lambda: gemm_attention(q, k, v)
            ref, got = sdpa().float(), gemm().float()
            err = ((got - ref).norm() / ref.norm()).item()
            t_sdpa, t_gemm = _time_it(sdpa), _time_it(gemm)
            del q, k, v, ref, got
            torch.cuda.empty_cache()
        ATTENTION[kind] = "gemm" if t_gemm < 0.95 * t_sdpa and err < 1e-2 else "sdpa"
    except Exception as e:
        ATTENTION[kind] = "sdpa"
        print(f"⚠️ {kind} attention benchmark failed ({e}); using SDPA")


benchmark_attention("dit", 1, 16384, 28, 128)  # 480p / 5 s, half the heads per GPU

# ==== VAE decode on both GPUs ====
# The video VAE decodes the latent in independent temporal clips (only the cross-fade between neighbours is
# sequential), so the clips are split between the two GPUs: GPU 1 uses the MMGP-managed decoder, GPU 0 a second
# decoder whose weights are copied, only for the duration of the decode, from the first one (no second copy in RAM;
# between videos it holds no weights at all). Tiling, attention and cross-fades are exactly Wan2GP's. On the first video GPU 0 also
# decodes clip 0 and must match GPU 1's result; otherwise decoding falls back to GPU 1 alone.
_vae_b = h3_main._load_video_vae(VIDEO_VAE_INT8_FILE, False)  # same architecture; its own weights are dropped below
_vae_b.encoder = _vae_b.quant_conv = None  # decode-only copy
_vae_b_parts = torch.nn.ModuleDict({"post_quant_conv": _vae_b.post_quant_conv, "decoder": _vae_b.decoder})
_vae_b_slots = []  # (module, "_parameters" | "_buffers", name, source module)
_source_modules = dict(pipeline.video_decoder.named_modules())
for _name, _module in _vae_b_parts.named_modules():
    for _store in ("_parameters", "_buffers"):
        for _key, _value in getattr(_module, _store).items():
            if _value is not None:
                _vae_b_slots.append((_module, _store, _key, _source_modules[_name]))


def _vae_b_attach(device):
    """Copy the first decoder's weights (pinned RAM, or GPU 1 if MMGP has it loaded) to `device`."""
    for module, store, key, source in _vae_b_slots:
        value = getattr(source, store)[key].to(device, non_blocking=True)
        if store == "_parameters":
            value = torch.nn.Parameter(value, requires_grad=False)
        getattr(module, store)[key] = value


def _vae_b_release():
    """Drop every weight of the second decoder. It must not keep references to the first decoder's tensors either:
    MMGP swaps in new GPU tensors when it loads a model, so a shared reference taken while the decoder is on GPU 1
    would keep those 2.3 GB of VRAM alive after MMGP unloads it."""
    for module, store, key, _ in _vae_b_slots:
        getattr(module, store)[key] = None


_vae_b_release()
gc.collect()

# Wan2GP decodes each clip as 256 px tiles stacked into ONE decoder batch: ~12 tiles at 480p (decode already peaks at
# ~13.3 GB of the T4's 14.6), ~28 at 720p, which does not fit next to half of the DiT + the LoRA. The decoder is a ViT
# that treats every tile on its own, so running the tiles in smaller groups gives exactly the same frames. The group
# size is planned per GPU from its free VRAM before each decode (~80 MB per tile measured, 100 MB budgeted), so the
# decoder never has to run out of memory; halving on an out-of-memory error stays only as a last resort.
VAE_TILE_GROUP = {}
VAE_TILE_BYTES = 100 * 2**20


def _tile_grouped(decoder):
    full_forward = decoder.forward

    def forward(hidden_states, *args, **kwargs):
        dev, tiles = hidden_states.device.index, hidden_states.shape[0]
        while True:
            n = min(VAE_TILE_GROUP.get(dev, tiles), tiles)
            try:
                if n >= tiles:
                    return full_forward(hidden_states, *args, **kwargs)
                return torch.cat([full_forward(hidden_states[i:i + n], *args, **kwargs) for i in range(0, tiles, n)])
            except torch.OutOfMemoryError:
                if n <= 1:
                    raise
                VAE_TILE_GROUP[dev] = max(1, n // 2)
                print(f"  ⚙️ VAE: GPU {dev} decodes {VAE_TILE_GROUP[dev]} tiles at a time from now on (out of memory with {n})")
                with torch.cuda.device(hidden_states.device):
                    torch.cuda.synchronize()  # kernels queued before the failed allocation may still use freed blocks
                    gc.collect()
                    torch.cuda.empty_cache()

    decoder.forward = forward


_tile_grouped(pipeline.vae.decoder)
_tile_grouped(_vae_b.decoder)


def _plan_tile_groups():
    """Tiles per decoder call that fit each GPU's free VRAM right now (GPU 1 loads its 2.3 GB decoder on first use)."""
    resident = any(p.is_cuda for p in pipeline.video_decoder.parameters())
    for dev in DEVICES:
        with torch.cuda.device(dev):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info(dev)[0]
        if dev == MAIN_DEV and not resident:
            free -= PINNED_BYTES.get("video decoder", 0)
        VAE_TILE_GROUP[dev.index] = max(1, int((free - 512 * 2**20) // VAE_TILE_BYTES))


# ==== Drop the checkpoint files from the page cache ====
# Loading reads ~40 GB of checkpoints, and Linux keeps whatever fits of them cached in RAM. That cache is free to
# reclaim (it never causes an out-of-memory restart), but Kaggle's RAM meter counts it, so the meter shows several GB
# more than the engine really uses. Everything is already pinned or on the GPUs, so the cached copies are useless.
# Pages a model still has memory-mapped are left alone by the kernel.
def drop_file_cache(*roots):
    for root in roots:
        for dirpath, _, files in os.walk(root, followlinks=True):
            for name in files:
                try:
                    fd = os.open(os.path.join(dirpath, name), os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                except (OSError, AttributeError):
                    pass


drop_file_cache(MODELS_DIR)
_orig_vae_decode = pipeline.vae._decode
PARALLEL_VAE = {"enabled": True, "verified": False}
PEAKS = {}


def _decode_clip_lean(model, z):
    """Wan2GP's _decode_clip (same 256 px tiles, same blend order and math) that decodes one row of tiles at a time and
    writes finished pixels straight into an FP16 CPU canvas. The original decodes every tile of the clip at once and
    builds the canvas on the GPU: at 2560x1440 that is ~1.1 GB of decoded tiles plus a 0.6 GB canvas, which do not fit
    next to the resident DiT halves and the decoder; here the GPU holds one row of tiles (~0.15 GB)."""
    if not model.use_tiling:
        return model.decoder(model.post_quant_conv(z)).to(torch.float16).cpu()
    ratio = model.spatial_compression_ratio
    height, width = z.shape[-2] * ratio, z.shape[-1] * ratio
    y_indices, y_lengths, y_overlaps = model._split_tiles(height, model.tile_sample_min_height,
                                                          model.tile_sample_min_overlap_height)
    x_indices, x_lengths, x_overlaps = model._split_tiles(width, model.tile_sample_min_width,
                                                          model.tile_sample_min_overlap_width)
    batch = z.shape[0]
    canvas, row_tails, out_y = None, [], 0
    for i, (i_pos, i_len) in enumerate(zip(y_indices, y_lengths)):
        tiles = [z[..., i_pos // ratio: i_pos // ratio + i_len // ratio, j_pos // ratio: j_pos // ratio + j_len // ratio]
                 for j_pos, j_len in zip(x_indices, x_lengths)]
        hidden = model.post_quant_conv(torch.cat(tiles, dim=0))
        del tiles
        decoded = model.decoder(hidden)
        del hidden
        new_tails, left_tail, out_x = [], None, 0
        for j in range(len(x_indices)):
            tile = decoded[j * batch:(j + 1) * batch]
            if i > 0:
                tile = model._blend(row_tails[j], tile, y_overlaps[i - 1], dim=-2)
            if j > 0:
                tile = model._blend(left_tail, tile, x_overlaps[j - 1], dim=-1)
            if i < len(y_indices) - 1:
                new_tails.append(tile[..., -y_overlaps[i]:, :].clone())
            left_tail = tile[..., :, -x_overlaps[j]:].clone() if j < len(x_indices) - 1 else None
            if i < len(y_indices) - 1:
                tile = tile[..., :-y_overlaps[i], :]
            if j < len(x_indices) - 1:
                tile = tile[..., :, :-x_overlaps[j]]
            if canvas is None:
                canvas = torch.empty(*tile.shape[:-2], height, width, dtype=torch.float16)
            canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
            out_x += tile.shape[-1]
        row_tails = new_tails
        out_y += tile.shape[-2]
        del decoded, tile
    return canvas


def parallel_vae_decode(z, sink=None):
    """Wan2GP's MiniMaxH3 _decode with the clips split between the two GPUs: clip 2k on GPU 1, clip 2k+1 on GPU 0, both
    at once, then stitched in order with the same cross-fades. With `sink`, every finished run of frames (float
    [1, 3, f, H, W] in [-1, 1], CPU) is handed over as soon as it is final and nothing else is kept, so a 2K video never
    sits in RAM as a whole (at 2560x1440 one decoded clip alone is ~0.6 GB). Without, the frames are returned as one
    FP16 tensor."""
    vae = pipeline.vae
    chunk_tokens, token_drop = vae.tokens_chunk_size, vae.config.token_drop
    ratio = vae.temporal_compression_ratio
    chunk_frames = chunk_tokens * ratio
    if z.shape[2] == 1:
        out = _orig_vae_decode(z)
        return sink(out) if sink is not None else out
    num_tokens = z.shape[2] + token_drop
    pad_tokens = (-num_tokens) % chunk_tokens
    num_chunks = (num_tokens + pad_tokens) // chunk_tokens - int(token_drop > 0)
    if num_chunks < 1:
        out = _orig_vae_decode(z)
        return sink(out) if sink is not None else out
    if pad_tokens > 0:
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
    intra_tail = vae.config.clip_length % ratio
    tokens_before_pad = z.shape[2] - pad_tokens
    pad_frames = sum(intra_tail if intra_tail and (tokens_before_pad + i) % chunk_tokens == 0 else ratio
                     for i in range(pad_tokens))
    output_frames = num_chunks * (chunk_frames - vae.frame_pre_padding) + vae.frame_overlap - pad_frames
    two_gpus = PARALLEL_VAE["enabled"] and num_chunks > 1

    _vae_b.use_tiling = vae.use_tiling
    for attr in ("tile_sample_min_height", "tile_sample_min_width", "tile_sample_min_overlap_height", "tile_sample_min_overlap_width"):
        setattr(_vae_b, attr, getattr(vae, attr))
    PEAKS["denoise"] = [torch.cuda.max_memory_allocated(d) for d in DEVICES]  # the DiT phase ends here
    for d in DEVICES:
        torch.cuda.reset_peak_memory_stats(d)
    BAR.start("decode", "VAE decoding", num_chunks, "clips")
    done = [0]

    def decode_one(dev, model, i):
        with torch.cuda.device(dev), torch.inference_mode():
            start = i * chunk_tokens
            return _decode_clip_lean(model, z[:, :, start:start + chunk_tokens + vae.token_overlap].to(dev))

    def run(jobs):
        """jobs: [(device, model, clip index)] decoded at the same time, one per GPU."""
        results, errors = [None] * len(jobs), []

        def work(n, dev, model, i):
            try:
                if not pipeline._interrupt:
                    results[n] = decode_one(dev, model, i)
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=work, args=(n, *job)) for n, job in enumerate(jobs[1:], 1)]
        for t in threads:
            t.start()
        work(0, *jobs[0])
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        if pipeline._interrupt:
            from models.minimax_h3.interrupt import GenerationInterrupted
            raise GenerationInterrupted
        return results

    decoded = [None]
    state = {"write": 0, "overlap": None}

    def emit(chunk):
        copy_frames = min(chunk.shape[2], output_frames - state["write"])
        if copy_frames <= 0:
            return
        chunk = vae._prepare_decoded_chunk(chunk[:, :, :copy_frames])
        if sink is not None:
            sink(chunk)
        else:
            if decoded[0] is None:
                decoded[0] = torch.empty(*chunk.shape[:2], output_frames, *chunk.shape[3:], dtype=torch.float16)
            decoded[0][:, :, state["write"]:state["write"] + copy_frames].copy_(chunk)
        state["write"] += copy_frames

    def stitch(clip):
        for j in range(int(token_drop > 0) + 1):
            chunk = clip[:, :, j * chunk_frames:(j + 1) * chunk_frames][:, :, vae.frame_pre_padding:].float()
            if j == 0:
                if state["overlap"] is not None:
                    chunk = vae._blend(state["overlap"], chunk, vae.frame_overlap, dim=-3)
                emit(chunk)
            else:
                state["overlap"] = chunk.contiguous()
            del chunk

    try:
        with torch.cuda.device(DIT_DEV):
            torch.cuda.empty_cache()
        if two_gpus:
            with torch.cuda.device(DIT_DEV):
                _vae_b_attach(DIT_DEV)
        _plan_tile_groups()

        def finished(clips):
            for clip in clips:
                stitch(clip)
                done[0] += 1
            BAR.update(done[0])  # once per round: both GPUs' clips finish together
            report(PROGRESS["decode_base"] + PROGRESS["decode_span"] * done[0] / num_chunks,
                   f"🎞️ VAE decoding {done[0]}/{num_chunks} clips")

        i = 0
        if two_gpus and not PARALLEL_VAE["verified"]:  # first video: GPU 0 also decodes clip 0, must match GPU 1
            check, ref = run([(MAIN_DEV, vae, 0), (DIT_DEV, _vae_b, 0)])[::-1]
            mismatch = ((check.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)).item()
            PARALLEL_VAE["verified"] = True
            del check
            if mismatch > 1e-2:
                PARALLEL_VAE["enabled"] = two_gpus = False
                print(f"⚠️ GPU 0's VAE copy disagrees with GPU 1 (rel diff {mismatch:.3f}); decoding on GPU 1 only from now on")
            else:
                pass
            finished([ref])
            del ref
            i = 1
        while i < num_chunks:
            jobs = [(MAIN_DEV, vae, i)] + ([(DIT_DEV, _vae_b, i + 1)] if two_gpus and i + 1 < num_chunks else [])
            finished(run(jobs))
            i += len(jobs)
        if state["overlap"] is not None and state["write"] < output_frames:
            emit(state["overlap"])
    finally:
        try:  # let every queued kernel finish before GPU 0's decoder weights are dropped and its memory released
            for dev in DEVICES:
                torch.cuda.synchronize(dev)
        finally:
            _vae_b_release()  # drop GPU 0's copy
            with torch.cuda.device(DIT_DEV):
                torch.cuda.empty_cache()
    if state["write"] != output_frames:
        raise RuntimeError(f"MiniMax H3 VAE decoded {state['write']} frames, expected {output_frames}")
    BAR.finish()
    return decoded[0]


pipeline.vae._decode = parallel_vae_decode
_orig_audio_decode = pipeline.audio_vae.decode


def _audio_decode(*args, **kwargs):
    BAR.start("audio", "Audio decoding")
    return _orig_audio_decode(*args, **kwargs)


pipeline.audio_vae.decode = _audio_decode


# ==== The two stages: preview (4-step distillation LoRA) and refine (Open-H3-Regenerate-2K) ====
# Both run on the same resident transformer and differ only in their LoRAs, kept on the GPUs as INT8 factors (0.2%
# error on their deltas). Each is applied live and exact: merging a LoRA into the INT8 weights does not work, because
# changes smaller than half an INT8 step round straight back to the old weights (measured on the real FL2VA weights:
# 0.1-25% of a distillation LoRA's effect survived). The refine LoRA is NVIDIA's merged rank-576 adapter (Open-H3
# refiner + LightX2V FL2VA Turbo v1.2 + LMS Sharp), ~1.9 GB per GPU even as INT8. It is loaded when an upscale starts
# (~45 s) and dropped again before the VAE decode: a 2K decode needs that VRAM next to the decoder, and previews keep it.
DISTILL_KEY = "⚡ PDMD 4-step"
REFINE_KEY = "🔍 Open-H3 Regenerate 2K"
LORA_PATHS = {DISTILL_KEY: PDMD_FILE, REFINE_KEY: REFINE_LORA_FILE}
REFINE_LORA_BYTES = int(0.62 * os.path.getsize(REFINE_LORA_FILE))  # per GPU: each shard keeps its half of B or A
lora_runtime.q8_keys.update(LORA_PATHS)


def stage_loras(stage):
    strengths = {DISTILL_KEY: 1.0} if stage == "preview" else {DISTILL_KEY: 0.0, REFINE_KEY: 1.0}
    return {key: (LORA_PATHS[key], strength) for key, strength in strengths.items()}


BAR.start("loras", "Loading the preview LoRA (INT8 on both GPUs)")
lora_runtime.apply(stage_loras("preview"))
BAR.finish()
drop_file_cache(LORA_DIR)


class SharpCurves:
    """LMS Sharp's 50 AdaLN curve branches, part of the Open-H3 refine recipe. Their input is Sharp's 8-D timestep curve,
    interpolated from the Ref2VA pruned AdaLN table at the model time t = 1 - sigma. That table differs from this FL2VA
    checkpoint's own table (by up to 0.14 per coordinate), so the branches cannot be folded into the AdaLN weights.
    Step 1 stores each branch as one [96768 x 8] matrix D = B @ A; while the refine stage runs, a forward hook adds
    curve(t) @ D^T to every block's AdaLN projection on both GPUs, exactly as Open-H3's CurveLinear does."""

    def __init__(self, path):
        from safetensors.torch import load_file

        data = load_file(path)
        self.table = data.pop("ref2va_adaln_t_table").float().to(MAIN_DEV)
        self.on = False
        self.coords = {}
        self.nbytes = 0
        for i, block in enumerate(transformer.blocks):
            delta = data.pop(f"blocks.{i}.delta").float()
            for s, dev in enumerate(DEVICES):
                linear = block.shards[s].adaln_proj.linear
                if tuple(delta.shape) != tuple(linear.weight.shape):
                    raise ValueError(f"Sharp curve branch {i} has shape {tuple(delta.shape)}, AdaLN is {tuple(linear.weight.shape)}")
                d = delta.to(dev)
                self.nbytes += d.numel() * 4
                linear.register_forward_hook(self._hook(d))
        if data:
            raise ValueError(f"Unexpected tensors in {os.path.basename(path)}: {sorted(data)[:4]}")
        original = transformer._time_embedding

        def time_embedding(timesteps, table=None):
            if self.on and table is None:
                t = timesteps.float().clamp(0.0, 1.0)
                curve = self.table.to(t.device)
                pos = t * (curve.shape[0] - 1)
                lower = pos.floor().long().clamp(max=curve.shape[0] - 2)
                coords = torch.lerp(curve[lower], curve[lower + 1], (pos - lower).unsqueeze(-1))
                self.coords = {coords.device: coords}
            return original(timesteps, table)

        transformer._time_embedding = time_embedding

    def _hook(self, delta):
        def hook(module, args, output):
            if not self.on:
                return output
            x = args[0]
            coords = self.coords.get(x.device)
            if coords is None:
                coords = self.coords[x.device] = next(iter(self.coords.values())).to(x.device)
            if coords.shape[0] != x.shape[0]:
                raise RuntimeError(f"Sharp curve: {coords.shape[0]} timesteps for {x.shape[0]} AdaLN rows")
            return output.add_((coords @ delta.t()).to(output.dtype))
        return hook


SHARP = SharpCurves(SHARP_FILE)


def set_stage(stage):
    """Switch the resident transformer between the preview and the refine recipe (LoRA strengths + Sharp curves)."""
    for key, (_, strength) in stage_loras(stage).items():
        lora_runtime.strength[key] = strength
    SHARP.on = stage == "refine"


# ==== Latent upscaler (LBH-123-AI 3D conv, the one Open-H3 uses) ====
# It works on the DiT's (normalized) latents, normalizes them once more with the VAE statistics and maps them to the
# target grid in one pass, exactly as Open-H3 calls it. It runs in FP16 on GPU 0 (Turing has no BF16 convolutions;
# FP32 would be ~5x slower), in temporal chunks with 6 frames of context on each side when the whole clip does not fit,
# and falls back to FP32 if FP16 ever produces a non-finite value.
from models.minimax_h3.latent_upscaler import MiniMaxH3LatentUpscaler
from models.minimax_h3.video_vae import LATENTS_MEAN as VIDEO_LATENTS_MEAN, LATENTS_STD as VIDEO_LATENTS_STD


def _load_upscaler():
    from safetensors.torch import load_file

    with torch.device("meta"):
        model = MiniMaxH3LatentUpscaler()
    state = {k: v.to(torch.float16) for k, v in load_file(UPSCALER_FILE).items()}
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False)


UPSCALER = _load_upscaler()
UPSCALE_CONTEXT = 6  # latent frames of temporal context on each side of a chunk


@torch.inference_mode()
def upscale_latent(latent, height, width, dtype=torch.float16):
    """[24, T, h, w] normalized latent (CPU, FP32) -> [24, T, height, width] on the target latent grid (CPU, FP32)."""
    dev = DIT_DEV
    t, h, w = latent.shape[1:]
    scale = (height / h + width / w) / 2.0
    mean = torch.tensor(VIDEO_LATENTS_MEAN).view(1, -1, 1, 1, 1)
    std = torch.tensor(VIDEO_LATENTS_STD).view(1, -1, 1, 1, 1)
    x = ((latent[None].float() - mean) / std)
    with torch.cuda.device(dev):
        torch.cuda.empty_cache()
        model = UPSCALER.to(dev, dtype)
        try:
            free = torch.cuda.mem_get_info(dev)[0] - 900 * 2**20
            per_frame = height * width * 512 * torch.finfo(dtype).bits // 8 * 6  # ~6 live 512-channel maps at full size
            chunk = t if per_frame * t <= free else max(4, int(free // per_frame) - 2 * UPSCALE_CONTEXT)
            out = torch.empty(1, 24, t, height, width)
            steps = [(a, min(t, a + chunk)) for a in range(0, t, chunk)]
            BAR.start("upscale", "Latent upscaler", len(steps), "chunks")
            for n, (a, b) in enumerate(steps, 1):
                lo, hi = max(0, a - UPSCALE_CONTEXT), min(t, b + UPSCALE_CONTEXT)
                seg = model._forward_segment(x[:, :, lo:hi].to(dev, dtype), scale, (hi - lo, height, width), None, lambda: None)
                if not bool(torch.isfinite(seg).all()):
                    raise FloatingPointError("non-finite latent upscaler output")
                out[:, :, a:b] = seg[:, :, a - lo:b - lo].float().cpu()
                del seg
                BAR.update(n)
                report(PROGRESS["fraction"], f"🔼 Latent upscaler {n}/{len(steps)}")
            BAR.finish()
        except FloatingPointError:
            BAR.abort()
            if dtype == torch.float32:
                raise
            print("  ⚠️ FP16 latent upscaler overflowed; running it again in FP32")
            UPSCALER.to("cpu", torch.float16)
            return upscale_latent(latent, height, width, torch.float32)
        finally:
            UPSCALER.to("cpu", torch.float16)
            torch.cuda.synchronize(dev)
            torch.cuda.empty_cache()
    return (out * std + mean)[0]


print(f"✅ Ready | RAM {_ram_in_use()}")
sys.stdout.flush()


def fetch_lora(entry_or_url):
    """Download a catalog LoRA or a direct .safetensors URL (Hugging Face 'resolve' links work) into LORA_DIR."""
    if isinstance(entry_or_url, dict):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=entry_or_url["repo"], filename=entry_or_url["file"], local_dir=LORA_DIR)
    url = entry_or_url.strip()
    m = re.match(r"https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/[^/]+/(.+?)(?:\?.*)?$", url)
    if m:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=m.group(1), filename=urllib.request.unquote(m.group(2)), local_dir=LORA_DIR)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(url.split("?")[0])) or "custom_lora.safetensors"
    if not name.endswith(".safetensors"):
        name += ".safetensors"
    path = os.path.join(LORA_DIR, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


# ==== Bilingual UI strings (Persian default) ====
CHANNEL = "AiWithMase"
YT_URL = "https://youtube.com/@AiWithMase"
TG_URL = "https://t.me/AiWithMase"

UI = {"lang": "fa"}


def set_lang(code):
    UI["lang"] = code if code in ("fa", "en") else "fa"


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


# ==== Prompt builder (MiniMax H3 prompt contract) ====
# "Persian" is not one of H3's 11 officially stable dialogue languages: it is supported to a varying degree.
LANGUAGES = ["English", "Chinese", "Spanish", "French", "German", "Japanese", "Korean", "Hindi", "Urdu", "Arabic",
             "Portuguese", "Russian", "Italian", "Turkish", "Indonesian", "Persian"]
_STRUCTURED = re.compile(r"(integrated_multimodal_description:|detailed_description:|subject_definitions:)")


def build_prompt(scene, dialogue, language, sound, music, duration_s, has_start, has_end, n_refs, ref_names, has_voice,
                 triggers):
    scene = (scene or "").strip()
    triggers = [t.strip() for t in triggers if t and t.strip().lower() not in scene.lower()]
    if _STRUCTURED.search(scene):  # the user wrote a full H3 prompt: keep it, only put trigger words in front
        return (" ".join(triggers) + " " + scene).strip() if triggers else scene
    # LoRA trigger words open the shot description, the way the LoRAs were captioned
    trigger = "".join(t + (" " if t[-1] in ".,;:" else ", ") for t in triggers)
    secs = f"{duration_s:.1f}".rstrip("0").rstrip(".")
    speaker = "<Subject 1>" if n_refs else "The main character"
    line = ""
    if dialogue and dialogue.strip():
        voice = " in the voice from <Audio 1>" if has_voice else ""
        line = (f" {speaker} speaks clearly{voice} with natural, synchronized lip movements and says "
                f"(S1) <d>[{language}] {dialogue.strip()}</d>")
    sound = (sound or "").strip() or "Natural ambient sound that matches the scene and the visible action."
    music = (music or "").strip() or "N/A"
    pictures = 0
    anchors = []
    if has_start:
        pictures += 1
        anchors.append(f"For the target video, at 0.00 seconds into the target video, <Picture {pictures}> (from [Shot 1]) is fully referenced.")
    if has_end:
        pictures += 1
        anchors.append(f"For the target video, at {duration_s:.2f} seconds into the target video, <Picture {pictures}> (from [Shot 1]) is fully referenced.")
    if not n_refs:
        lead = "Continue directly from <Picture 1>, preserving its subject, composition and lighting. " if has_start else ""
        body = (f"integrated_multimodal_description: [Shot 1] {trigger}A {secs}-second cinematic single take. {lead}{scene}{line}\n"
                f"overall_soundscape: {sound}\nnon_diegetic_music: {music}")
        return "\n".join(anchors + [body])
    names = [n.strip() for n in (ref_names or "").split(",") if n.strip()]
    subjects, retention = [], []
    for i in range(n_refs):
        label = names[i] if i < len(names) else ("the main subject" if i == 0 else "the subject")
        picture = pictures + i + 1
        subjects.append(f"<Subject {i + 1}> is {label} in <Picture {picture}>, preserving their exact identity, "
                        f"appearance, proportions, clothing and distinctive details.")
        retention.append(f"<Subject {i + 1}> (appears in [Shot 1]): fully_preserved - identity and appearance are retained; "
                         f"the setting, action, camera and lighting are new.")
    if has_voice:
        retention.append("<Audio 1>: reference - the voice timbre and delivery of <Subject 1> follow <Audio 1>.")
    task = "[reference generation + audio reference]" if has_voice else "[reference generation]"
    return "\n".join(anchors + [
        "subject_definitions:", *subjects,
        "summary:", f"{task} {scene}",
        "retention_analysis:", *retention,
        "detailed_description:",
        f"{trigger}The target video is a {secs}-second cinematic single take. [Shot 1] {scene}{line} Identities, motion and "
        f"lighting remain stable and physically coherent.",
        f"overall_soundscape: {sound}",
        f"non_diegetic_music: {music}",
    ])


# ==== Sizes, durations and a VRAM guard ====
from PIL import Image
from models.minimax_h3.interrupt import GenerationInterrupted
from models.minimax_h3.transformer import VISUAL_COND_TIMESTEP, patchify_video

PREVIEW_AREA = 672 * 384  # Open-H3's smallest documented draft size (384P): every preview is made at this pixel area
ASPECTS = {"16:9 Landscape": 16 / 9, "9:16 Portrait": 9 / 16, "1:1 Square": 1.0, "4:3 Classic": 4 / 3,
           "3:4 Portrait": 3 / 4, "21:9 Cinema": 21 / 9, "Auto (match image)": None}
DURATIONS = {"3 s": 73, "4.5 s": 107, "5 s": 124, "6.5 s": 158, "8 s": 192, "10 s": 243, "12 s": 294, "15 s": 362}
PREVIEW_ONLY = "384p preview only"
OUTPUTS = {"720p": 720, "1080p": 1080, "2K (1440p)": 1440}
REFINE_PRESETS = {  # refine steps, tile area range (px), tile overlap, and whether the grid is picked for speed
    "⚡ Fast - 2 steps, slim overlaps": dict(steps=2, max_area=750_000, min_area=450_000, overlap=0.125, fastest=True),
    "⚖️ Balanced - 3 steps": dict(steps=3, max_area=750_000, min_area=0, overlap=0.25, fastest=False),
    "💎 Quality - 3 steps, bigger tiles": dict(steps=3, max_area=1_250_000, min_area=0, overlap=0.25, fastest=False),
}
DEFAULT_PRESET = "⚡ Fast - 2 steps, slim overlaps"
AUDIO_LATENT_FPS = h3_pipeline.AUDIO_LATENT_FPS


def _shifted_grid(start, steps, shift=6.0):
    """Open-H3's refine sigma grid: even steps in unshifted sigma from `start`, then the shift-6 map."""
    raw = start / (shift - (shift - 1) * start)
    return [start] + [shift * x / (1 + (shift - 1) * x) for x in (raw * (1 - i / steps) for i in range(1, steps))] + [0.0]


# Euler, CFG 1, model time 1 - sigma. 3 steps is the release setting; 2 steps starts from less noise (softer texture).
SCHEDULES = {3: [0.909375, 0.8571428571428571, 0.6666666666666666, 0.0], 2: _shifted_grid(0.7, 2)}


def preview_size(aspect, image):
    ratio = ASPECTS.get(aspect)
    if ratio is None:
        ratio = image.width / image.height if image is not None else 16 / 9
    w = max(256, round(math.sqrt(PREVIEW_AREA * ratio) / 32) * 32)
    h = max(256, round(PREVIEW_AREA / w / 32) * 32)
    return w, h, ratio


def output_size(ratio, short):
    """((width, height) delivered, (width, height) computed). The computed grid is rounded up to whole 32 px latent
    patches (1080 -> 1088, 720 -> 736) and the centre is kept, as Open-H3 does for 1080P."""
    if ratio >= 1:
        out_w, out_h = round(short * ratio / 2) * 2, short
    else:
        out_w, out_h = short, round(short / ratio / 2) * 2
    return (out_w, out_h), (math.ceil(out_w / 32) * 32, math.ceil(out_h / 32) * 32)


def estimate_rows(w, h, frames, images_in_prompt):
    per_frame = (h // 32) * (w // 32)
    video = h3_pipeline.video_latent_frames(frames) * per_frame
    audio = round(frames / 24 * 40) * 2
    text = 1200 + images_in_prompt * per_frame  # prompt tokens + Qwen vision tokens of each picture
    return video + audio + images_in_prompt * per_frame + text


def free_for_denoise():
    aux_offload.unload_all()
    gc.collect()
    frees = []
    for dev in DEVICES:
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
        frees.append(torch.cuda.mem_get_info(dev)[0])
    return frees


def needed_bytes(rows, long_video=False):
    """Working VRAM per GPU: activations per packed row (half the heads per GPU) + scratch. Estimates."""
    return rows * (48 if long_video else 110) * 1024 + 700 * 1024**2


def refine_bytes(rows, long_video=True):
    """As needed_bytes, plus the rank-576 refine LoRA's FP16 temporaries (8192-row slices, dequantized factors)."""
    return needed_bytes(rows, long_video) + 500 * 1024**2


# ==== Spatial tiles for the regenerate pass ====
# A 5 s 2K clip is ~134k tokens: one full-frame attention pass would take ~30 min per step on two T4s, and its
# activations alone exceed a T4. So the frame is regenerated as overlapping tiles (25% overlap, 32 px aligned). Each
# tile keeps its true position in the full frame (RoPE ids from the whole canvas, Wan2GP's target_spatial_context),
# and every step the tiles' clean-latent predictions are blended with sin^2 / cos^2 ramps over the overlaps (weights sum
# to 1 everywhere) before the shared Euler update, so neighbouring tiles agree. Attention cost falls with the square of
# the tile size; the preset sets the largest tile area and the overlap (25%, or 12.5% for Fast: ~15% fewer tokens per
# tile, never under 64 px), and the planner uses the fewest tiles within it that fit the VRAM.
MIN_TILE = 256
MIN_OVERLAP = 64


def axis_tiles(length, n, overlap=0.25):
    """n overlapping tiles along one axis, in pixels, 32 px aligned: [(start, size)]."""
    if n == 1:
        return [(0, length)]
    size = min(length, math.ceil(length / (n - (n - 1) * overlap) / 32) * 32)
    return [(round(k * (length - size) / (n - 1) / 32) * 32, size) for k in range(n)]


def tile_rows(th, tw, latent_t, text_len, audio_t, keyframes):
    per_frame = (th // 32) * (tw // 32)
    return (latent_t + keyframes) * per_frame + 2 * audio_t + text_len


def plan_tiles(height, width, latent_t, text_len, audio_t, keyframes, max_area, free, overlap=0.25, min_area=0,
               fastest=False):
    """(rows tiles, column tiles, packed rows per tile, long-video mode): among grids whose tile area is within
    [min_area, max_area], whose tile sides stay within 2:1 (thin strips see too little of the scene) and whose
    activations fit `free` bytes, the fewest tiles (or, with `fastest`, the lowest estimated time); squarer tiles break
    ties. None if nothing fits."""
    best = None
    for nr in range(1, 9):
        for nc in range(1, 9):
            ys, xs = axis_tiles(height, nr, overlap), axis_tiles(width, nc, overlap)
            th, tw = ys[0][1], xs[0][1]
            if (nr > 1 and th >= height) or (nc > 1 and tw >= width) or min(th, tw) < min(MIN_TILE, height, width):
                continue
            if any(a + size - b < MIN_OVERLAP for axis in (ys, xs) for (a, size), (b, _) in zip(axis, axis[1:])):
                continue
            rows = tile_rows(th, tw, latent_t, text_len, audio_t, keyframes)
            if refine_bytes(rows, True) > free:
                continue
            aspect = max(th / tw, tw / th)
            cost = nr * nc * step_seconds(rows) if fastest else nr * nc
            key = (not min_area <= th * tw <= max_area, aspect > 2.0, cost, aspect)
            if best is None or key < best[0]:
                best = (key, ys, xs, rows, refine_bytes(rows, False) > free)
    return None if best is None else best[1:]


def preview_step_seconds(rows):
    """Seconds per preview step on the tensor-parallel T4 pair, x = rows / 16k (Kaggle: 20.7 s at ~10.3k rows)."""
    x = rows / 16384
    return 24.0 * x * x + 7.0 * x + 6.0


def step_seconds(rows):
    """Seconds per upscale tile pass: the preview fit x 1.23 for the rank-576 LoRA and the low-VRAM slices
    (Kaggle: 98.7 s at 26.5k rows)."""
    return 1.23 * preview_step_seconds(rows)


def decode_seconds(w, h, frames):
    return 0.6e-6 * w * h * frames + 5.0  # two-GPU VAE decode (~55 s for 832x480 x 124 frames on one GPU)


REFINE_FREE = [0]  # free VRAM per GPU right after start-up (the planner's budget for estimates)


def upscale_plan(ratio, frames, target, preset, keyframes=0, text_len=800, free=None):
    (ow, oh), (cw, ch) = output_size(ratio, OUTPUTS[target])
    p = REFINE_PRESETS[preset]
    steps = p["steps"]
    latent_t = h3_pipeline.video_latent_frames(frames)
    audio_t = round(frames / 24 * AUDIO_LATENT_FPS)
    plan = plan_tiles(ch, cw, latent_t, text_len, audio_t, keyframes, p["max_area"], free or REFINE_FREE[0],
                      p["overlap"], p["min_area"], p["fastest"])
    if plan is None:
        return None
    ys, xs, rows, long_video = plan
    seconds = steps * len(ys) * len(xs) * step_seconds(rows) + decode_seconds(cw, ch, frames) + 50  # + LoRA load, upscaler
    return dict(out=(ow, oh), comp=(cw, ch), steps=steps, ys=ys, xs=xs, rows=rows, long_video=long_video,
                seconds=seconds, latent_t=latent_t, audio_t=audio_t)


def eta_text(output, preset, duration, aspect):
    frames = DURATIONS[duration]
    ratio = ASPECTS.get(aspect) or 16 / 9
    w, h, _ = preview_size(aspect, None)
    preview_s = 12 + PREVIEW_STEPS * preview_step_seconds(estimate_rows(w, h, frames, 0) - 600) + decode_seconds(w, h, frames)
    text = t("eta.head", w=w, h=h, time=_fmt_secs(preview_s))
    if output in OUTPUTS:
        plan = upscale_plan(ratio, frames, output, preset)
        if plan is None:
            return text + t("eta.nofit", output=output, duration=duration)
        tiles = len(plan["ys"]) * len(plan["xs"])
        text += t("eta.upscale", output=output, w=plan["out"][0], h=plan["out"][1],
                  time=_fmt_secs(plan["seconds"]), tiles=tiles, steps=plan["steps"])
    return text


# ==== Live progress (also polled by the page on the Cloudflare link, which holds back Gradio's SSE stream) ====
PROGRESS = {"active": False, "fraction": 0.0, "text": "", "started": 0.0, "finished": 0.0, "video": None, "message": "",
            "decode_base": 0.86, "decode_span": 0.11, "final_n": 0}


def report(fraction, text):
    PROGRESS.update(active=True, fraction=float(fraction), text=text)


def _to_video_tensor(image):
    return h3_pipeline._pil_to_video(image.convert("RGB"))


def _ffmpeg_cmd(w, h, wav, path, fps=24):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if wav:
        cmd += ["-i", wav, "-c:a", "aac", "-b:a", "192k"]
    return cmd + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                  "-shortest", path]


@torch.inference_mode()  # the decoded video is an inference tensor
def save_mp4(video, wav, path, fps=24):
    """Streams [3, T, H, W] video in [-1, 1] to ffmpeg a few frames at a time: no full-size copies in RAM."""
    _, t, h, w = video.shape
    proc = subprocess.Popen(_ffmpeg_cmd(w, h, wav, path, fps), stdin=subprocess.PIPE)
    try:
        for start in range(0, t, 16):
            frames = video[:, start:start + 16].to(torch.float32, copy=True).clamp_(-1, 1).add_(1).mul_(127.5).round_()
            proc.stdin.write(frames.to(torch.uint8).permute(1, 2, 3, 0).contiguous().numpy().tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode}")


@torch.inference_mode()
def decode_to_mp4(latent, wav, path, out_w, out_h, frames):
    """Normalized [24, T, H, W] latent -> MP4 with the preview's audio. The two-GPU VAE decode hands finished frames
    straight to ffmpeg (centre-cropped to the delivered size), so the video never exists in RAM as a whole."""
    vae = pipeline.vae
    pipeline._configure_tiling(None)
    mean = torch.tensor(VIDEO_LATENTS_MEAN).view(1, -1, 1, 1, 1)
    std = torch.tensor(VIDEO_LATENTS_STD).view(1, -1, 1, 1, 1)
    z = (latent[None].float() * std + mean).to(vae._model_dtype)
    height, width = latent.shape[-2] * 16, latent.shape[-1] * 16
    top, left = (height - out_h) // 2, (width - out_w) // 2
    proc = subprocess.Popen(_ffmpeg_cmd(out_w, out_h, wav, path), stdin=subprocess.PIPE)
    written = [0]

    def sink(chunk):
        n = min(chunk.shape[2], frames - written[0])
        if n <= 0:
            return
        pixels = chunk[0, :, :n, top:top + out_h, left:left + out_w]
        pixels = pixels.float().clamp_(-1, 1).add_(1).mul_(127.5).round_().to(torch.uint8)
        proc.stdin.write(pixels.permute(1, 2, 3, 0).contiguous().numpy().tobytes())
        written[0] += n

    try:
        parallel_vae_decode(z, sink)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    except BaseException:
        proc.kill()
        raise
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode}")
    return written[0]


# ==== Preview capture: the final latent and the prompt states are kept so a preview can be upscaled later ====
CAPTURE = {}
_orig_encode_prompt = pipeline._encode_prompt


def _encode_prompt_capture(prompt, presentation):
    context, text_tags = _orig_encode_prompt(prompt, presentation)
    CAPTURE["context"] = context.detach().to("cpu", copy=True)
    CAPTURE["text_tags"] = torch.as_tensor(text_tags).detach().to("cpu", copy=True)
    return context, text_tags


pipeline._encode_prompt = _encode_prompt_capture


@torch.inference_mode()
def run_generation(prompt_text, frames, w, h, steps, sampler, shift, seed, start_img=None, end_img=None):
    pipeline._interrupt = False
    t0 = time.time()
    timings = {"text": 0.0, "denoise": 0.0, "decode": 0.0}
    phase = {"name": "prep", "t": time.time(), "step_t": time.time()}

    def callback(step, latent=None, is_start=False, override_num_inference_steps=None, progress_unit=None,
                 progress_title=None, **kwargs):
        now = time.time()
        if progress_unit == "layers":  # Qwen3-VL text / image encoding
            if phase["name"] != "text":
                phase.update(name="text", t=now)
            total = override_num_inference_steps or 50
            BAR.start("text", "Prompt encoding", total, "layers")
            BAR.update(max(0, step + 1), total)
            report(0.02 + 0.13 * min(1.0, (step + 1) / total), f"📝 {progress_title or 'Encoding prompt'} {max(0, step + 1)}/{total}")
            return
        if progress_unit == "tiles":  # VAE encode / decode
            if "Decod" in (progress_title or "") and phase["name"] != "decode":
                if phase["name"] == "denoise":
                    timings["denoise"] = now - phase["t"]
                phase.update(name="decode", t=now)
            total = override_num_inference_steps or 1
            decoding = "Decod" in (progress_title or "")
            BAR.start("decode" if decoding else "encode", "VAE decoding" if decoding else "Image encoding", total, "tiles")
            BAR.update(max(0, step + 1), total)
            report(0.86 + 0.11 * min(1.0, (step + 1) / total), f"🎞️ {progress_title or 'VAE'} {max(0, step + 1)}/{total}")
            return
        if is_start:
            if phase["name"] == "text":
                timings["text"] = now - phase["t"]
            phase.update(name="denoise", t=now, step_t=now, total=override_num_inference_steps or steps)
            BAR.start("denoise", "Denoising", phase["total"], "steps")
            report(0.15, f"🎬 Preview: denoising step 0/{phase['total']}")
            return
        total = phase.get("total", steps)
        dt = now - phase["step_t"]
        phase["step_t"] = now
        done = step + 1
        if done >= total and latent is not None:  # the finished FP32 latent, before the VAE's lower-precision cast
            CAPTURE["latent"] = latent.detach().float().to("cpu", copy=True)
        BAR.update(done, total)
        report(0.15 + 0.70 * done / total, f"🎬 Preview: denoising step {done}/{total} ({dt:.1f}s/it)")

    kwargs = dict(input_prompt=prompt_text, frame_num=frames, height=h, width=w, sampling_steps=int(steps), seed=int(seed),
                  shift=float(shift), sample_solver=sampler, fps=24, guide_phases=1, callback=callback,
                  image_refs_relative_size=100, video_prompt_type="", audio_prompt_type="")
    if start_img is not None:
        kwargs["image_start"] = _to_video_tensor(start_img)
    if end_img is not None:
        kwargs["image_end"] = _to_video_tensor(end_img)
    report(0.01, "⚡ Preparing...")
    BAR.start("prep", "Preparing inputs")
    PROGRESS.update(decode_base=0.86, decode_span=0.11)
    try:
        result = pipeline.generate(**kwargs)
    finally:
        BAR.finish()
    if result is None:
        return None, None, None, timings
    if phase["name"] == "decode":
        timings["decode"] = time.time() - phase["t"]
    elif phase["name"] == "denoise":  # decoding reports through the parallel decoder, not the step callback
        timings["denoise"] = phase["step_t"] - phase["t"]
        timings["decode"] = time.time() - phase["step_t"]
    timings["total"] = time.time() - t0
    return result["x"], result.get("audio"), result.get("audio_sampling_rate", 32000), timings


# ==== Saved previews ====
def _preview_files(name):
    base = os.path.join(PREVIEW_DIR, name)
    return base + ".pt", base + ".json"


def list_previews():
    """[(label, name)], newest first."""
    import json

    items = []
    for meta_path in glob.glob(os.path.join(PREVIEW_DIR, "*.json")):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            items.append((os.path.getmtime(meta_path), meta["label"], meta["name"]))
        except Exception:
            pass
    return [(label, name) for _, label, name in sorted(items, reverse=True)]


def preview_meta(name):
    import json

    with open(_preview_files(name)[1]) as f:
        return json.load(f)


def _cleanup():
    BAR.abort()
    for key in _tp_cache:  # last hidden state / rope / temb copies held on both GPUs
        _tp_cache[key] = None
    gc.collect()
    for dev in DEVICES:
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()


def _user_loras(lora1, s1, lora2, s2, custom_url, custom_file, s3, auto_trigger):
    """{key: (path, strength)} for the LoRA pickers + the trigger words to put in the prompt."""
    selections, triggers = {}, []
    for choice, strength in ((lora1, s1), (lora2, s2)):
        if choice and choice != NO_LORA and float(strength) != 0.0 and choice not in selections:
            entry = LORA_CATALOG[choice]
            selections[choice] = (fetch_lora(entry), float(strength))
            if auto_trigger and entry["trigger"]:
                triggers.append(entry["trigger"])
    custom = None
    if custom_file is not None:
        custom = custom_file if isinstance(custom_file, str) else custom_file.name
    elif custom_url and custom_url.strip():
        custom = fetch_lora(custom_url)
    if custom and float(s3) != 0.0:
        selections[f"custom:{os.path.basename(custom)}"] = (custom, float(s3))
    return selections, triggers


# ==== Stage 1: preview ====
def on_main_gpu(fn):
    """CUDA's current device is per thread: Gradio workers start on GPU 0, which is full of DiT blocks.
    Run every handler with GPU 1 current, the device the sampler, text encoder and VAEs use for "cuda"."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with torch.cuda.device(MAIN_DEV):
            return fn(*args, **kwargs)
    return wrapped


@on_main_gpu
def make_preview(prompt, dialogue, language, sound, music, start_img, end_img, aspect, duration, seed, lora1, s1, lora2,
                 s2, custom_url, custom_file, s3, auto_trigger, sampler, shift, custom_steps):
    """Returns (preview MP4 path or None, status, preview name or None)."""
    import json
    import soundfile as sf

    try:
        if not (prompt or "").strip() and start_img is None:
            return None, tr("❌ اول یک پرامپت بنویس.", "❌ Please write a prompt."), None
        frames = DURATIONS[duration]
        steps = int(custom_steps) if custom_steps and int(custom_steps) > 0 else PREVIEW_STEPS
        w, h, ratio = preview_size(aspect, start_img if start_img is not None else end_img)
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)

        report(0.01, "🎨 Preparing LoRAs...")
        user, triggers = _user_loras(lora1, s1, lora2, s2, custom_url, custom_file, s3, auto_trigger)
        lora_runtime.apply({**stage_loras("preview"), **user})
        set_stage("preview")

        images_in_prompt = (start_img is not None) + (end_img is not None)
        rows = estimate_rows(w, h, frames, images_in_prompt)
        frees = free_for_denoise()
        LONG_VIDEO["on"] = needed_bytes(rows) > min(frees)
        if needed_bytes(rows, LONG_VIDEO["on"]) > min(frees):
            return None, (f"❌ {w}x{h} for {duration} needs about {_gb(needed_bytes(rows, True))} of working VRAM per GPU, "
                          f"but only {_gb(frees[0])} (GPU 0) / {_gb(frees[1])} (GPU 1) is free. Pick a shorter duration."), None

        build = lambda has_start, has_end: build_prompt(prompt, dialogue, language, sound, music, frames / 24, has_start,
                                                       has_end, 0, "", False, triggers)
        final_prompt = build(start_img is not None, end_img is not None)
        mode = ("First + Last Frame → Video" if start_img is not None and end_img is not None
                else "Image → Video" if start_img is not None else "Last Frame → Video" if end_img is not None else "Text → Video")
        print(f"\n{'=' * 70}\n🎬 Preview | {mode} | {w}x{h} | {frames / 24:.1f}s | {steps} steps | seed {seed}")
        if user:
            print("🎨 LoRAs: " + ", ".join(f"{k} @ {v[1]:.2f}" for k, v in user.items()))
        print(f"📝 {final_prompt[:300]}{'...' if len(final_prompt) > 300 else ''}\n{'=' * 70}")
        sys.stdout.flush()

        for dev in DEVICES:
            torch.cuda.reset_peak_memory_stats(dev)
        PEAKS.clear()
        CAPTURE.clear()
        video, audio, sr, timings = run_generation(final_prompt, frames, w, h, steps, sampler, shift, seed, start_img, end_img)
        if video is None:
            return None, "⏹️ Generation stopped.", None
        if "latent" not in CAPTURE or "context" not in CAPTURE:
            raise RuntimeError("the preview latent or prompt states were not captured")

        report(0.97, "💾 Saving MP4...")
        name = f"h3_preview_{time.strftime('%Y%m%d_%H%M%S')}_seed{seed}"
        pt_path, meta_path = _preview_files(name)
        wav = os.path.join(PREVIEW_DIR, name + ".wav")
        sf.write(wav, np.asarray(audio, dtype=np.float32), int(sr))
        out_path = os.path.join(OUTPUT_DIR, name + ".mp4")
        BAR.start("save", "Saving MP4")
        save_mp4(video, wav, out_path)
        BAR.finish()
        images = []
        for tag, img in (("start", start_img), ("end", end_img)):
            if img is None:
                images.append(None)
                continue
            path = os.path.join(PREVIEW_DIR, f"{name}_{tag}.png")
            img.convert("RGB").save(path)
            images.append(path)
        torch.save({"latent": CAPTURE["latent"], "context": CAPTURE["context"], "text_tags": CAPTURE["text_tags"],
                    "prompt": final_prompt, "text_prompt": build(False, False), "frames": frames, "w": w, "h": h,
                    "ratio": ratio, "seed": seed, "loras": user, "images": images, "wav": wav, "video": out_path}, pt_path)
        label = f"{time.strftime('%H:%M:%S')} · {(prompt or mode).strip()[:48]} · {w}x{h} · {frames / 24:.1f}s · seed {seed}"
        with open(meta_path, "w") as f:
            json.dump({"name": name, "label": label, "frames": frames, "ratio": ratio, "w": w, "h": h,
                       "keyframes": images_in_prompt}, f)
        total = timings.get("total", 0.0)
        per_step = timings["denoise"] / max(1, steps)
        print(f"✅ Preview saved in {total:.0f}s (prompt {timings['text']:.0f}s + denoise {timings['denoise']:.0f}s + "
              f"VAE/audio {timings['decode']:.0f}s): {out_path}")
        drop_file_cache(LORA_DIR, OUTPUT_DIR)
        sys.stdout.flush()
        status = "\n".join([
            tr(f"✅ پیش‌نمایش در {total:.0f} ثانیه ساخته شد ({mode} · {w}×{h} · {frames / 24:.1f} ثانیه · سید {seed})",
               f"✅ Preview done in {total:.0f}s ({mode}, {w}x{h}, {frames / 24:.1f}s, seed {seed})"),
            tr(f"⚡ پرامپت {timings['text']:.0f}s | دی‌نویز {timings['denoise']:.0f}s ({per_step:.1f}s/استپ) | VAE و صدا {timings['decode']:.0f}s",
               f"⚡ Prompt {timings['text']:.0f}s | Denoising {timings['denoise']:.0f}s ({per_step:.1f}s/step) | VAE + audio {timings['decode']:.0f}s"),
            tr(f"🎨 LoRAها: {', '.join(f'{k} @ {v[1]:.2f}' for k, v in user.items()) or 'هیچ'}",
               f"🎨 LoRAs: {', '.join(f'{k} @ {v[1]:.2f}' for k, v in user.items()) or 'none'}"),
            f"📁 {out_path}",
        ])
        return out_path, status, name
    except torch.OutOfMemoryError:
        traceback.print_exc()
        return None, tr("❌ حافظهٔ گرافیک پر شد. مدت کوتاه‌تری انتخاب کن.", "❌ Out of GPU memory. Pick a shorter duration and try again."), None
    except Exception as e:
        traceback.print_exc()
        if "CUDA error" in str(e):  # e.g. an illegal memory access: the GPU context is unusable from here on
            print("❌ The GPUs are in an error state. Re-run Step 2 to restart the engine (the models reload in ~4 min).")
            return None, f"❌ GPU error: {e}\nRe-run Step 2 to restart the engine.", None
        return None, f"❌ Error: {e}", None
    finally:
        _cleanup()


# ==== Stage 2: upscale + regenerate (Open-H3-Regenerate-2K) ====
SILENCE = {}


@torch.inference_mode()
def silence_latents(frames, audio_t):
    """Encoded stereo silence for the audio stream, as in Open-H3: the refiner LoRA was trained with it (clean, t = 1).
    The preview's own audio track is muxed into the final video unchanged."""
    if audio_t not in SILENCE:
        samples = round(frames / 24 * h3_pipeline.AUDIO_SAMPLE_RATE)
        latent = pipeline._encode_audio(torch.zeros(1, 2, samples)).float().cpu()  # [1, 32, 2, A]
        if latent.shape[-1] < audio_t:
            latent = torch.cat([latent, latent[..., -1:].expand(*latent.shape[:-1], audio_t - latent.shape[-1])], dim=-1)
        SILENCE[audio_t] = latent[..., :audio_t].contiguous()
    return SILENCE[audio_t]


def _crop(tensor, top, left, height, width):
    return tensor[..., top:top + height, left:left + width]


def regenerate_tiles(canvas, tiles, weight_sum, sigmas, forward, on_pass=None):
    """Euler from sigmas[0] to 0 over a tiled canvas, in place. Every step each tile's clean-latent prediction
    x0 = x_t + sigma * velocity (H3's sign convention) is weighted, accumulated, normalized by the summed weights and
    used for the one shared update x <- x0 + (sigma_next / sigma) * (x - x0)."""
    for step in range(len(sigmas) - 1):
        sigma, following = float(sigmas[step]), float(sigmas[step + 1])
        sigma_t = torch.tensor([sigma])
        predicted = torch.zeros_like(canvas)
        for n, (box, weights, payload) in enumerate(tiles, 1):
            if pipeline._interrupt:
                raise GenerationInterrupted
            x = _crop(canvas, *box)
            velocity = forward(x, sigma_t, payload)
            clean = velocity.float().mul_(sigma).add_(x.to(velocity.device))
            _crop(predicted, *box).add_(clean.cpu().mul_(weights))
            del x, velocity, clean
            if on_pass is not None:
                on_pass(step, n, step * len(tiles) + n)
        predicted.div_(weight_sum)
        ratio = following / sigma
        canvas.mul_(ratio).add_(predicted, alpha=1.0 - ratio)
        del predicted
    return canvas


@on_main_gpu
def make_upscale(name, target, preset, refine_seed, anchor):
    """Returns (MP4 path or None, status)."""
    t0 = time.time()
    try:
        if not name:
            return None, t("no.preview")
        if target not in OUTPUTS:
            return None, t("pick.output")
        pt_path, _ = _preview_files(name)
        if not os.path.exists(pt_path):
            return None, f"❌ Preview {name} was not found in {PREVIEW_DIR}."
        rec = torch.load(pt_path, map_location="cpu", weights_only=False)
        frames, latent = rec["frames"], rec["latent"]
        images = [Image.open(p).convert("RGB") if p and os.path.exists(p) else None for p in rec["images"]]
        use_keyframes = bool(anchor) and any(img is not None for img in images)
        n_kf = sum(img is not None for img in images) if use_keyframes else 0
        seed = int(refine_seed) if refine_seed is not None and int(refine_seed) >= 0 else rec["seed"]
        plan = upscale_plan(rec["ratio"], frames, target, preset, n_kf, int(rec["text_tags"].numel()))
        if plan is None:
            return None, (f"❌ {target} for a {frames / 24:.1f}s clip does not fit the VRAM even in small tiles. "
                          f"Make a shorter preview.")
        (ow, oh), (cw, ch), steps = plan["out"], plan["comp"], plan["steps"]
        sigmas = SCHEDULES[steps]
        lat_h, lat_w, audio_t = ch // 16, cw // 16, plan["audio_t"]
        pipeline._interrupt = False
        pipeline._configure_tiling(None)  # VAE tiles for the keyframe / silence encodes (also after an engine restart)
        PROGRESS.update(decode_base=0.90, decode_span=0.08)
        print(f"\n{'=' * 70}\n🔍 Upscale {rec['w']}x{rec['h']} → {ow}x{oh} ({target}) | {frames} frames | {steps} steps | "
              f"seed {seed}{' | start/end image re-anchored' if use_keyframes else ''}")

        report(0.01, "🎨 Loading the Open-H3 refine LoRA (~45 s)...")
        user = {k: v for k, v in (rec.get("loras") or {}).items() if os.path.exists(v[0])}
        BAR.start("loras", "Loading the refine LoRA")
        lora_runtime.apply({**stage_loras("refine"), **user})
        BAR.finish()
        set_stage("refine")
        with torch.inference_mode():
            if any(img is not None for img in images) and not use_keyframes:
                # without the pictures in the layout, the prompt must not refer to them either
                report(0.02, "📝 Encoding the text-only prompt...")
                context_raw, text_tags = _orig_encode_prompt(rec["text_prompt"], [])
                context_raw, text_tags = context_raw.detach().cpu(), torch.as_tensor(text_tags).cpu()
            else:
                context_raw, text_tags = rec["context"], rec["text_tags"]
            keyframe_latents, keyframes = [], []
            if use_keyframes:  # the start / end image at the target resolution, as clean-ish keyframe conditions
                report(0.03, "🖼️ Encoding the start / end image at the target resolution...")
                for index, img in enumerate(images):
                    if img is not None:
                        video = h3_pipeline._resize_video(_to_video_tensor(img), ch, cw)[:, :1]
                        keyframe_latents.append(pipeline._encode_video(video).float())
                        keyframes.append({"anchor": "first" if index == 0 else "last", "latent_frame_count": 1})
            silence = silence_latents(frames, audio_t)
            report(0.05, "🔼 Latent upscaler...")
            t_up = time.time()
            upscaled = upscale_latent(latent, lat_h, lat_w)
            t_up = time.time() - t_up
            frees = free_for_denoise()
            report(0.08, "📝 Refining the prompt states...")
            context = transformer.preprocess_text_embeds(context_raw.to(MAIN_DEV))
            plan = upscale_plan(rec["ratio"], frames, target, preset, n_kf, int(text_tags.numel()), min(frees))
            if plan is None:
                return None, (f"❌ Not enough free VRAM for {target} right now ({_gb(frees[0])} / {_gb(frees[1])} free). "
                              f"Remove extra LoRAs or make a shorter preview.")
            ys, xs = plan["ys"], plan["xs"]
            LONG_VIDEO["on"] = plan["long_video"]
            print(f"🧩 {len(ys)}x{len(xs)} tiles of {xs[0][1]}x{ys[0][1]} px | estimate {_fmt_secs(plan['seconds'])}")
            sys.stdout.flush()

            generator = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn((1, *upscaled.shape), generator=generator, dtype=torch.float32)
            conditions = [lat.mul(VISUAL_COND_TIMESTEP).add_(torch.randn(lat.shape, generator=generator), alpha=1.0 - VISUAL_COND_TIMESTEP)
                          for lat in keyframe_latents]
            canvas = torch.lerp(upscaled[None], noise, float(sigmas[0]))  # (1 - s0) * upscaled + s0 * noise
            del noise, upscaled
            tiles, weight_sum = [], torch.zeros(1, 1, 1, lat_h, lat_w)
            for r, (top, th) in enumerate(ys):
                top_ov = 0 if r == 0 else ys[r - 1][0] + ys[r - 1][1] - top
                bottom_ov = 0 if r == len(ys) - 1 else top + th - ys[r + 1][0]
                for c, (left, tw) in enumerate(xs):
                    left_ov = 0 if c == 0 else xs[c - 1][0] + xs[c - 1][1] - left
                    right_ov = 0 if c == len(xs) - 1 else left + tw - xs[c + 1][0]
                    box = (top // 16, left // 16, th // 16, tw // 16)
                    weights = h3_pipeline._phase_2_tile_weights(box[2], box[3], top_ov // 16, bottom_ov // 16,
                                                                left_ov // 16, right_ov // 16)
                    _crop(weight_sum, *box).add_(weights)
                    cond_rows = (torch.cat([patchify_video(_crop(cond, *box).contiguous(), transformer.patch_size)
                                            for cond in conditions]) if conditions else None)
                    payload = {"keyframes": keyframes or None, "audio_keyframes": None, "refs": None,
                               "cond_video_rows": cond_rows, "cond_audio_rows": None, "frame_count": frames,
                               "text_token_tags": text_tags, "fps": 24, "target_audio_condition_latents": 0,
                               "target_video_condition_frames": 0, "attention_sparsity": 1.0,
                               "target_spatial_context": (lat_h, lat_w, box[0], box[1])}
                    tiles.append((box, weights, payload))
            del conditions, keyframe_latents
            if float(weight_sum.min()) <= 0:
                raise RuntimeError("tile weights do not cover the frame")

            audio = silence.to(MAIN_DEV)
            audio_sigma = torch.zeros(1, device=MAIN_DEV)
            total = steps * len(tiles)
            BAR.start("refine", "Regenerating", total, "tile passes")
            t_dn = time.time()

            def forward(x, sigma_t, payload):
                velocity, _ = transformer(x.to(MAIN_DEV), audio, sigma_t.to(MAIN_DEV), audio_sigma, context, payload)
                return velocity

            def on_pass(step, n, done):
                BAR.update(done)
                report(0.10 + 0.78 * done / total,
                       f"🔍 Regenerating {target}: step {step + 1}/{steps}, tile {n}/{len(tiles)} · "
                       f"~{_fmt_secs((time.time() - t_dn) / done * (total - done))} left")

            regenerate_tiles(canvas, tiles, weight_sum, sigmas, forward, on_pass)
            BAR.finish()
            t_dn = time.time() - t_dn
            refined = canvas[0]
            canvas = tiles = context = audio = None  # the DiT phase's tensors go before the VAE loads
        set_stage("preview")
        lora_runtime.apply(stage_loras("preview"))  # frees the refine LoRA's VRAM for the decoder
        _cleanup()

        report(0.90, "🎞️ VAE decoding...")
        out_path = os.path.join(OUTPUT_DIR, f"{name.replace('h3_preview', 'h3_' + target.split()[0].lower())}.mp4")
        for dev in DEVICES:
            torch.cuda.reset_peak_memory_stats(dev)
        PEAKS.clear()
        t_dec = time.time()
        decode_to_mp4(refined, rec["wav"], out_path, ow, oh, frames)
        t_dec = time.time() - t_dec
        total = time.time() - t0
        per_pass = t_dn / max(1, steps * len(ys) * len(xs))
        print(f"✅ {target} saved in {_fmt_secs(total)} (regenerate {_fmt_secs(t_dn)} + VAE/MP4 {_fmt_secs(t_dec)}): "
              f"{out_path}\n")
        drop_file_cache(LORA_DIR, OUTPUT_DIR, PREVIEW_DIR)
        sys.stdout.flush()
        return out_path, "\n".join([
            tr(f"✅ {target} در {_fmt_secs(total)} آماده شد ({ow}×{oh} · {frames / 24:.1f} ثانیه · سید بازسازی {seed})",
               f"✅ {target} done in {_fmt_secs(total)} ({ow}x{oh}, {frames / 24:.1f}s, refine seed {seed})"),
            f"⚡ Latent upscaler {t_up:.0f}s | Regenerate {_fmt_secs(t_dn)} ({len(ys)}x{len(xs)} tiles x {steps} steps, "
            f"{per_pass:.1f}s per tile pass) | VAE + MP4 {_fmt_secs(t_dec)}",
            "🔊 " + (tr("صدا: ندارد (ویدیوی تو صدا ندارد)", "Audio: none (your video has no sound)") if not rec["wav"] else
                     tr("صدا: صدای ویدیوی تو", "Audio: your video's soundtrack") if rec.get("source") == "upload" else
                     tr("صدا: صدای پیش‌نمایش", "Audio: the preview's soundtrack")),
            f"📁 {out_path}",
        ])
    except GenerationInterrupted:
        return None, tr("⏹️ آپاسکیل متوقف شد.", "⏹️ Upscale stopped.")
    except torch.OutOfMemoryError:
        traceback.print_exc()
        return None, tr("❌ حافظهٔ گرافیک پر شد. پریست ⚡ سریع یا ⚖️ متعادل، خروجی پایین‌تر یا پیش‌نمایش کوتاه‌تر را امتحان کن.",
                        "❌ Out of GPU memory. Try the ⚡ Fast or ⚖️ Balanced preset, a lower output or a shorter preview.")
    except Exception as e:
        traceback.print_exc()
        if "CUDA error" in str(e):
            print("❌ The GPUs are in an error state. Re-run Step 2 to restart the engine (the models reload in ~4 min).")
            return None, tr(f"❌ خطای GPU: {e}\nدوباره Step 2 را اجرا کن تا موتور بالا بیاید.",
                            f"❌ GPU error: {e}\nRe-run Step 2 to restart the engine.")
        return None, tr(f"❌ خطا: {e}", f"❌ Error: {e}")
    finally:
        set_stage("preview")
        try:
            lora_runtime.apply(stage_loras("preview"))  # drops the preview's extra LoRAs again
        except Exception:
            traceback.print_exc()
        _cleanup()


# ==== Your own video as the starting point ====
# Instead of a preview, an uploaded clip is conformed and encoded by the H3 VAE, and that latent goes through the same
# upscale. Open-H3 prepares drafts from other video models this way: 24 fps, cover-scaled and centre-cropped to 576P,
# cut or padded (last frame held), encoded with a fixed posterior sample. The clip's own soundtrack is kept.
UPLOAD_AREA = 1024 * 576


def _probe(path):
    """(display width, display height, seconds, has audio) of a video file."""
    import json

    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height:stream_tags=rotate:stream_side_data=rotation:format=duration",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("no video stream in the uploaded file")
    width, height = int(video["width"]), int(video["height"])
    rotation = int(float((video.get("tags") or {}).get("rotate", 0) or 0))
    for side in video.get("side_data_list", []) or []:
        rotation = int(float(side.get("rotation", rotation) or rotation))
    if abs(rotation) % 180 == 90:  # phone clips: ffmpeg rotates them on decode
        width, height = height, width
    seconds = float((info.get("format") or {}).get("duration") or 0.0)
    return width, height, seconds, any(s.get("codec_type") == "audio" for s in streams)


def conform_upload(path, frames, name):
    """The clip at 24 fps, cover-scaled and centre-cropped to the encode size of its aspect ratio, cut or padded to
    `frames`: (uint8 frames [F, H, W, 3], width, height, aspect ratio, wav or None, conformed MP4 with the audio)."""
    w0, h0, _, has_audio = _probe(path)
    ratio = w0 / h0
    w = max(256, round(math.sqrt(UPLOAD_AREA * ratio) / 32) * 32)
    h = max(256, round(UPLOAD_AREA / w / 32) * 32)
    vf = (f"fps=24,scale={w}:{h}:force_original_aspect_ratio=increase:flags=bicubic,crop={w}:{h},"
          "tpad=stop_mode=clone:stop_duration=60")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-frames:v", str(frames), "-an",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True).stdout
    if len(raw) != frames * h * w * 3:
        raise ValueError(f"could not read {frames} frames from the uploaded video")
    pixels = np.frombuffer(raw, np.uint8).reshape(frames, h, w, 3)
    wav = None
    if has_audio:
        wav = os.path.join(PREVIEW_DIR, name + ".wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vn", "-ac", "2", "-ar", "32000", "-af", "apad",
                        "-t", f"{frames / 24:.4f}", wav], check=True)
    mp4 = os.path.join(OUTPUT_DIR, name + ".mp4")  # what gets encoded: the "before" of the comparison
    proc = subprocess.Popen(_ffmpeg_cmd(w, h, wav, mp4), stdin=subprocess.PIPE)
    proc.stdin.write(pixels.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode}")
    return pixels, w, h, ratio, wav, mp4


@torch.inference_mode()
def encode_upload(pixels):
    """uint8 [F, H, W, 3] -> normalized latent [24, T, H/16, W/16]. Wan2GP's encode_condition, one 17-frame clip at a
    time on the GPU (a 10 s 576P clip as one tensor would be ~2 GB of VRAM): same last-frame padding, same dropped
    trailing latents, same fixed-seed posterior sample."""
    vae = pipeline.vae
    pipeline._configure_tiling(None)
    clip = vae.config.clip_length
    frames = pixels.shape[0]
    starts = list(range(0, frames + (-frames) % clip, clip))
    BAR.start("encode", "VAE encoding your video", len(starts), "clips")
    moments = []
    for n, start in enumerate(starts, 1):
        if pipeline._interrupt:
            raise GenerationInterrupted
        index = [min(i, frames - 1) for i in range(start, start + clip)]
        chunk = torch.from_numpy(pixels[index]).to(MAIN_DEV).permute(3, 0, 1, 2)[None].float().div_(127.5).sub_(1.0)
        moments.append(vae._encode_clip(vae._pixels(chunk)).float().cpu())
        del chunk
        BAR.update(n)
        report(0.02 + 0.06 * n / len(starts), f"🎞️ Encoding your video {n}/{len(starts)}")
    BAR.finish()
    moments = torch.cat(moments, dim=2)
    if vae.config.token_drop > 0:
        moments = moments[:, :, :-vae.config.token_drop]
    mean, logvar = moments.chunk(2, dim=1)
    std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
    noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(42), dtype=torch.float32)
    latents = (mean + std * noise).to(torch.float16).float()
    mean_t = torch.tensor(VIDEO_LATENTS_MEAN).view(1, -1, 1, 1, 1)
    std_t = torch.tensor(VIDEO_LATENTS_STD).view(1, -1, 1, 1, 1)
    return ((latents - mean_t) / std_t)[0]


def upload_frames(seconds, duration):
    """Frames to use: the chosen duration, or the longest duration the clip covers (shorter clips hold their last frame)."""
    wanted = DURATIONS[duration]
    available = int(seconds * 24 + 0.5)
    if available >= wanted:
        return wanted
    fitting = [f for f in sorted(DURATIONS.values()) if f <= available]
    return fitting[-1] if fitting else min(DURATIONS.values())


@on_main_gpu
def make_upload(video, prompt, dialogue, language, sound, music, duration, seed, lora1, s1, lora2, s2, custom_url,
                custom_file, s3, auto_trigger):
    """An uploaded clip -> an upscalable record, like a preview. Returns (conformed MP4 or None, status, name or None)."""
    import json

    t0 = time.time()
    try:
        if not video:
            return None, t("no.video"), None
        path = video if isinstance(video, str) else video.name
        _, _, seconds, _ = _probe(path)
        frames = upload_frames(seconds, duration)
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)
        name = f"h3_preview_{time.strftime('%Y%m%d_%H%M%S')}_upload"
        pipeline._interrupt = False
        report(0.01, "🎞️ Reading your video...")
        pixels, w, h, ratio, wav, mp4 = conform_upload(path, frames, name)
        print(f"\n{'=' * 70}\n🎞️ Your video | {os.path.basename(path)} | {w}x{h} | {frames / 24:.1f}s | "
              f"{'with' if wav else 'no'} audio | seed {seed}")
        user, triggers = _user_loras(lora1, s1, lora2, s2, custom_url, custom_file, s3, auto_trigger)
        scene = (prompt or "").strip() or "A cinematic video with natural detail and lighting."
        final_prompt = build_prompt(scene, dialogue, language, sound, music, frames / 24, False, False, 0, "", False, triggers)
        print(f"📝 {final_prompt[:300]}{'...' if len(final_prompt) > 300 else ''}\n{'=' * 70}")
        sys.stdout.flush()
        latent = encode_upload(pixels)
        del pixels
        report(0.09, "📝 Encoding the prompt...")
        BAR.start("text", "Prompt encoding")
        with torch.inference_mode():
            context, text_tags = _orig_encode_prompt(final_prompt, [])
            context, text_tags = context.detach().to("cpu", copy=True), torch.as_tensor(text_tags).to("cpu", copy=True)
        BAR.finish()
        if latent.shape[1] != h3_pipeline.video_latent_frames(frames):
            raise RuntimeError(f"encoded {latent.shape[1]} latent frames, expected {h3_pipeline.video_latent_frames(frames)}")
        pt_path, meta_path = _preview_files(name)
        torch.save({"latent": latent, "context": context, "text_tags": text_tags, "prompt": final_prompt,
                    "text_prompt": final_prompt, "frames": frames, "w": w, "h": h, "ratio": ratio, "seed": seed,
                    "loras": user, "images": [None, None], "wav": wav, "video": mp4, "source": "upload"}, pt_path)
        label = f"{time.strftime('%H:%M:%S')} · 🎞️ {os.path.basename(path)[:40]} · {w}x{h} · {frames / 24:.1f}s"
        with open(meta_path, "w") as f:
            json.dump({"name": name, "label": label, "frames": frames, "ratio": ratio, "w": w, "h": h, "keyframes": 0,
                       "source": "upload"}, f)
        print(f"✅ Your video encoded in {_fmt_secs(time.time() - t0)}: {mp4}")
        sys.stdout.flush()
        return mp4, f"✅ Your video is ready to upscale ({w}x{h}, {frames / 24:.1f}s, {'with' if wav else 'no'} audio)", name
    except GenerationInterrupted:
        return None, "⏹️ Stopped.", None
    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Could not use this video: {e}", None
    finally:
        free_for_denoise()
        _cleanup()


# ==== Before / after comparison player ====
# Both videos stacked in one frame, the preview clipped at a draggable divider; PROGRESS_HEAD's script wires it up
# (gr.HTML does not run scripts). The upscaled video is the clock and carries the sound; the preview follows it.
TARGET_TOKENS = {"720p": "720p", "1080p": "1080p", "2k": "2K (1440p)"}  # file-name token -> output label


def _file_url(path):
    import urllib.parse
    return "gradio_api/file=" + urllib.parse.quote(path, safe="/")  # Gradio's file route (supports seeking)


def compare_html(preview_path, final_path, target, out_w, out_h, comp_w, comp_h, before="384p preview"):
    if not (preview_path and final_path and os.path.exists(preview_path) and os.path.exists(final_path)):
        return compare_idle()
    # The latent upscaler maps the whole preview onto the computed grid (e.g. 1280x736) and the delivered video is its
    # centre crop (1280x720): stretch the preview over that grid and shift it so the two line up pixel for pixel.
    width, height = comp_w / out_w * 100, comp_h / out_h * 100
    left, top = -(comp_w - out_w) / 2 / out_w * 100, -(comp_h - out_h) / 2 / out_h * 100
    before_style = f"left:{left:.4f}%;top:{top:.4f}%;width:{width:.4f}%;height:{height:.4f}%;object-fit:fill"
    return f"""
<div class="aiq-compare" tabindex="0" data-fps="24">
  <div class="aiq-cmp-stage" style="aspect-ratio:{out_w}/{out_h}">
    <video class="aiq-after" src="{_file_url(final_path)}" preload="auto" playsinline></video>
    <div class="aiq-cmp-clip"><video class="aiq-before" src="{_file_url(preview_path)}" preload="auto" muted
      playsinline style="{before_style}"></video></div>
    <div class="aiq-cmp-line"><div class="aiq-cmp-knob">⟷</div></div>
    <span class="aiq-tag aiq-tag-l">{t("compare.before")} · {before}</span>
    <span class="aiq-tag aiq-tag-r">{t("compare.after")} · Open-H3 {target.split()[0]}</span>
  </div>
  <div class="aiq-cmp-bar">
    <button class="aiq-btn aiq-play" type="button">{t("compare.play")}</button>
    <button class="aiq-btn aiq-prev" type="button" title="{t("compare.prev")}">⏮</button>
    <button class="aiq-btn aiq-next" type="button" title="{t("compare.next")}">⏭</button>
    <input class="aiq-seek" type="range" min="0" max="1000" step="1" value="0">
    <span class="aiq-time">0:00 / 0:00</span>
    <button class="aiq-btn aiq-loop" type="button" title="{t("compare.loop")}">🔁</button>
    <button class="aiq-btn aiq-mute" type="button" title="{t("compare.sound")}">🔊</button>
    <button class="aiq-btn aiq-full" type="button" title="{t("compare.full")}">⛶</button>
  </div>
  <div class="aiq-cmp-hint">{t("compare.hint")} · {out_w}×{out_h}</div>
</div>"""


def compare_for(final_path):
    """The comparison for an upscaled MP4 (h3_<target>_<stamp>.mp4), from its preview h3_preview_<stamp>."""
    m = re.match(r"h3_(720p|1080p|2k)_(.+)\.mp4$", os.path.basename(final_path or ""))
    if not m:
        return compare_idle()
    name = "h3_preview_" + m.group(2)
    try:
        meta = preview_meta(name)
    except Exception:
        return compare_idle()
    target = TARGET_TOKENS[m.group(1)]
    (ow, oh), (cw, ch) = output_size(meta["ratio"], OUTPUTS[target])
    before = t("compare.yourvideo") if meta.get("source") == "upload" else t("compare.preview")
    return compare_html(os.path.join(OUTPUT_DIR, name + ".mp4"), final_path, target, ow, oh, cw, ch, before)


# ==== Generation as a background job ====
# A 2K upscale takes 20-40 minutes, longer than the Gradio share / Cloudflare tunnels reliably keep one event stream
# open. So the buttons only start a job and return; the page asks for the job's state every 2 s with short,
# independent requests (queue=False: plain HTTP, no event stream), and a reloaded page picks the results up again.
JOB = {"id": 0, "done": 0, "running": False, "preview": None, "preview_n": 0, "final": None, "final_n": 0,
       "preview_name": None, "compare": compare_idle(), "status": "Ready."}


def _set_preview(path, name):
    JOB.update(preview=path, preview_n=JOB["preview_n"] + 1, preview_name=name)


def _set_final(path):
    JOB.update(final=path, final_n=JOB["final_n"] + 1, compare=compare_for(path))
    PROGRESS["final_n"] = JOB["final_n"]  # the page script fetches the new comparison player when this changes


def _generate_job(args):
    *preview_args, output, preset, refine_seed, anchor = args
    path, status, name = make_preview(*preview_args)
    if path:
        _set_preview(path, name)
    if not path or output not in OUTPUTS:
        return path, status
    JOB["status"] = status
    final, up_status = make_upscale(name, output, preset, refine_seed, anchor)
    if final:
        _set_final(final)
    return final or path, status + "\n\n" + up_status


def _upscale_job(args):
    name, output, preset, refine_seed, anchor = args
    final, status = make_upscale(name, output, preset, refine_seed, anchor)
    if final:
        _set_final(final)
    return final, status


def _upload_job(args):
    *upload_args, output, preset, refine_seed = args
    path, status, name = make_upload(*upload_args)
    if path:
        _set_preview(path, name)
    if not path:
        return path, status
    JOB["status"] = status
    final, up_status = make_upscale(name, output, preset, refine_seed, False)
    if final:
        _set_final(final)
    return final or path, status + "\n\n" + up_status


def _job_worker(job_id, fn, args):
    try:
        video, status = fn(args)
    except BaseException as e:  # the stages report their own errors; this is a last resort
        traceback.print_exc()
        video, status = None, f"❌ Error: {e}"
    JOB.update(running=False, done=job_id, status=status)
    PROGRESS.update(active=False, finished=time.time(), video=os.path.basename(video) if video else None,
                    message=status.splitlines()[0] if status else "")


def _start(fn, args, message):
    if JOB["running"]:
        return t("job.running")
    JOB.update(id=JOB["id"] + 1, running=True)
    PROGRESS.update(active=True, fraction=0.0, text=t("starting"), started=time.time(), finished=0.0, video=None, message="")
    threading.Thread(target=_job_worker, args=(JOB["id"], fn, args), daemon=True).start()
    return message


def start_generation(*args):
    return _start(_generate_job, args, t("start.gen"))


def start_upscale(name, output, preset, refine_seed, anchor):
    if output not in OUTPUTS:
        return t("pick.output")
    if not name:
        return t("no.preview")
    return _start(_upscale_job, (name, output, preset, refine_seed, anchor), t("start.upscale", output=output))


def start_upload(video, prompt, dialogue, language, sound, music, duration, seed, lora1, s1, lora2, s2, custom_url,
                 custom_file, s3, auto_trigger, output, preset, refine_seed):
    if not video:
        return t("no.video")
    if output not in OUTPUTS:
        return t("pick.output")
    args = (video, prompt, dialogue, language, sound, music, duration, seed, lora1, s1, lora2, s2, custom_url, custom_file,
            s3, auto_trigger, output, preset, refine_seed)
    return _start(_upload_job, args, t("start.upload", output=output))


def poll_job(seen):
    """Every 2 s: live progress while a job runs, each new preview / final video once per page."""
    seen = list(seen or [0, 0, 0])
    preview = final = picker = gr.skip()
    if JOB["preview_n"] != seen[0]:
        seen[0] = JOB["preview_n"]
        preview = JOB["preview"]
        picker = gr.update(choices=list_previews(), value=JOB["preview_name"])
    if JOB["final_n"] != seen[1]:
        seen[1] = JOB["final_n"]
        final = JOB["final"]
    if JOB["running"]:
        elapsed = time.time() - PROGRESS["started"] if PROGRESS["started"] else 0.0
        status = f"⏳ {PROGRESS['text'] or t('working')} · {_fmt_secs(elapsed)}"
    elif JOB["done"] and JOB["done"] != seen[2]:
        seen[2] = JOB["done"]
        status = JOB["status"]
    else:
        status = gr.skip()
    return preview, final, status, seen, picker


def deliver_compare():
    """The comparison player, once per finished upscale. Not a gr.Timer output: Gradio re-renders a timer's HTML outputs
    on every tick (the two videos reloaded and the player blinked), so the page script clicks a hidden button instead."""
    return JOB["compare"]


def stop_generation():
    if not JOB["running"]:
        return t("stop.none")
    pipeline._interrupt = True
    return t("stop.stopping")


def load_latest_video():
    videos = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")), key=os.path.getmtime)
    if not videos:
        return gr.skip(), gr.skip(), t("nothing.yet", path=OUTPUT_DIR), gr.skip()
    latest = videos[-1]
    if "_preview_" in os.path.basename(latest):
        return latest, gr.skip(), t("loaded.preview", path=latest), gr.skip()
    return gr.skip(), latest, t("loaded.video", path=latest), compare_for(latest)


def refresh_previews():
    choices = list_previews()
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def lora_info(choice):
    if not choice or choice == NO_LORA:
        return gr.update(value=1.0), ""
    entry = LORA_CATALOG[choice]
    trig = f" Trigger: `{entry['trigger']}`" if entry["trigger"] else ""
    return gr.update(value=entry["scale"]), f"ℹ️ {entry['info']}{trig}"


def picker_eta(name, output, preset):
    if not name or output not in OUTPUTS:
        return ""
    try:
        meta = preview_meta(name)
    except Exception:
        return ""
    plan = upscale_plan(meta["ratio"], meta["frames"], output, preset, meta.get("keyframes", 0))
    if plan is None:
        return t("picker.nofit", output=output, duration=f"{meta['frames'] / 24:.1f}")
    return t("picker.eta", output=output, w=plan["out"][0], h=plan["out"][1], time=_fmt_secs(plan["seconds"]),
             tiles=len(plan["ys"]) * len(plan["xs"]), steps=plan["steps"])


PROGRESS_HEAD = """
<script>
(function () {
  // Progress + result overlay fed by plain HTTP polling: it keeps working when a tunnel drops Gradio's event stream.
  var box = null, fill = null, label = null, hideAt = 0, lastFinal = 0;
  function tt(fa, en) { return window.__aiqLang === 'en' ? en : fa; }
  function build() {
    box = document.createElement("div");
    box.style.cssText = "position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:99999;width:min(560px,92vw);" +
      "background:#1f2937;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.25);padding:10px 14px;display:none;" +
      "font-family:Vazirmatn,Inter,sans-serif;color:#fff;font-size:13px";
    label = document.createElement("div");
    label.style.cssText = "margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis";
    var track = document.createElement("div");
    track.style.cssText = "height:8px;background:rgba(255,255,255,.18);border-radius:6px;overflow:hidden";
    fill = document.createElement("div");
    fill.style.cssText = "height:100%;width:0;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);transition:width .6s ease";
    track.appendChild(fill); box.appendChild(label); box.appendChild(track); document.body.appendChild(box);
  }
  async function tick() {
    try {
      var r = await fetch("aiq_progress?t=" + Date.now(), {cache: "no-store"});
      var p = await r.json();
      var deliver = document.getElementById("aiq-deliver");
      if (deliver && p.final_n && p.final_n !== lastFinal) { lastFinal = p.final_n; deliver.click(); }
      if (!box && document.body) build();
      if (box) {
        if (p.active) {
          box.style.display = "block"; hideAt = 0;
          fill.style.width = Math.round(p.fraction * 100) + "%";
          label.textContent = "⏳ " + p.text + " · " + Math.round(p.elapsed) + "s";
        } else if (box.style.display === "block") {
          if (!hideAt) {
            fill.style.width = "100%";
            label.textContent = (p.video ? tt("✅ تمام شد در ", "✅ Done in ") : tt("⚠️ پایان یافت پس از ", "⚠️ Finished after ")) + Math.round(p.elapsed) + "s ";
            if (p.video) {
              var link = document.createElement("a");
              link.href = "aiq_video?t=" + Date.now(); link.target = "_blank";
              link.style.cssText = "color:#c7d2fe;font-weight:600";
              link.textContent = tt("· ▶ باز کردن / دانلود ویدیو", "· ▶ open / download the video");
              label.appendChild(link);
            } else if (p.message) { label.textContent += "· " + p.message; }
            hideAt = Date.now() + 60000;
          } else if (Date.now() > hideAt) { box.style.display = "none"; }
        }
      }
    } catch (e) {}
    setTimeout(tick, 1000);
  }
  tick();
})();
</script>
<script>
(function () {
  function fmt(t) {
    if (!isFinite(t)) t = 0;
    return Math.floor(t / 60) + ":" + String(Math.floor(t % 60)).padStart(2, "0");
  }
  function init(root) {
    root.dataset.ready = "1";
    var stage = root.querySelector(".aiq-cmp-stage"), clip = root.querySelector(".aiq-cmp-clip");
    var line = root.querySelector(".aiq-cmp-line"), after = root.querySelector("video.aiq-after");
    var before = root.querySelector("video.aiq-before"), play = root.querySelector(".aiq-play");
    var seek = root.querySelector(".aiq-seek"), time = root.querySelector(".aiq-time");
    var loop = root.querySelector(".aiq-loop"), mute = root.querySelector(".aiq-mute");
    var fps = parseFloat(root.dataset.fps) || 30, dragging = false, seeking = false;
    before.muted = true;
    function setPos(p) {
      p = Math.max(0, Math.min(100, p));
      clip.style.clipPath = "inset(0 " + (100 - p) + "% 0 0)";
      line.style.left = p + "%";
    }
    function fromEvent(e) {
      var r = stage.getBoundingClientRect();
      return (e.clientX - r.left) / r.width * 100;
    }
    function align() { if (Math.abs(before.currentTime - after.currentTime) > 0.001) before.currentTime = after.currentTime; }
    function showState() { play.textContent = after.paused ? tt("▶ پخش", "▶ Play") : tt("⏸ مکث", "⏸ Pause"); }
    function toggle() {
      if (after.paused) {
        if (after.ended || after.currentTime >= (after.duration || 0) - 0.01) { after.currentTime = 0; }
        align();
        var a = after.play(), b = before.play();
        if (a && a.catch) a.catch(function () {});
        if (b && b.catch) b.catch(function () {});
      } else {
        after.pause(); before.pause(); align();
      }
    }
    function step(dir) {
      after.pause(); before.pause();
      var t = Math.max(0, Math.min((after.duration || 0), after.currentTime + dir / fps));
      after.currentTime = t; before.currentTime = t;
    }
    setPos(50);
    stage.addEventListener("pointerdown", function (e) {
      dragging = true; stage.setPointerCapture(e.pointerId); setPos(fromEvent(e)); e.preventDefault();
      root.focus({ preventScroll: true });
    });
    stage.addEventListener("pointermove", function (e) { if (dragging) setPos(fromEvent(e)); });
    ["pointerup", "pointercancel"].forEach(function (n) { stage.addEventListener(n, function () { dragging = false; }); });
    play.addEventListener("click", toggle);
    root.querySelector(".aiq-prev").addEventListener("click", function () { step(-1); });
    root.querySelector(".aiq-next").addEventListener("click", function () { step(1); });
    loop.addEventListener("click", function () {
      after.loop = before.loop = !after.loop; loop.classList.toggle("aiq-off", !after.loop);
    });
    mute.addEventListener("click", function () {
      after.muted = !after.muted; mute.textContent = after.muted ? "🔇" : "🔊";
    });
    root.querySelector(".aiq-full").addEventListener("click", function () {
      if (document.fullscreenElement) { document.exitFullscreen(); }
      else if (stage.requestFullscreen) { stage.requestFullscreen(); }
    });
    seek.addEventListener("input", function () {
      seeking = true;
      var t = (after.duration || 0) * seek.value / 1000;
      after.currentTime = t; before.currentTime = t;
    });
    seek.addEventListener("change", function () { seeking = false; });
    after.addEventListener("play", showState);
    after.addEventListener("pause", function () { before.pause(); align(); showState(); });
    after.addEventListener("ended", function () { before.pause(); showState(); });
    after.addEventListener("seeked", align);
    root.addEventListener("keydown", function (e) {
      if (e.code === "Space") { toggle(); e.preventDefault(); }
      else if (e.code === "ArrowLeft") { step(-1); e.preventDefault(); }
      else if (e.code === "ArrowRight") { step(1); e.preventDefault(); }
    });
    loop.classList.add("aiq-off");
    (function tick() {
      if (!root.isConnected) return;
      if (!after.paused && !before.paused && Math.abs(before.currentTime - after.currentTime) > 0.1) align();
      if (!seeking && after.duration) seek.value = Math.round(after.currentTime / after.duration * 1000);
      var label = fmt(after.currentTime) + " / " + fmt(after.duration);
      if (time.textContent !== label) time.textContent = label;
      requestAnimationFrame(tick);
    })();
  }
  setInterval(function () {
    document.querySelectorAll(".aiq-compare:not([data-ready])").forEach(init);
  }, 400);
})();
</script>
"""

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
(lang) => {
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


# ==== Cloudflare quick tunnel: a second public link in case the Gradio share link fails ====
# Started without preexec_fn: that would force a real fork() of this process (Python otherwise uses vfork), and a
# fork has to copy every pinned page at once (see _dont_fork). Step 2 stops leftover tunnels before each start.
# The link only works once (a) cloudflared has registered a connection with Cloudflare's edge and (b) the random
# *.trycloudflare.com name is published in DNS. cloudflared tries QUIC (UDP) first, which Kaggle's network often blocks,
# and only falls back to HTTP/2 after a while: hence --protocol http2. And a browser that asks for the name before it
# exists caches the "not found" answer for minutes, so the link is shown as ready only after both checks pass.
def _dns_published(host):
    """Ask Cloudflare's DNS-over-HTTPS directly: no local / Kaggle resolver caches a negative answer.
    True / False, or None when the lookup itself is unavailable."""
    try:
        req = urllib.request.Request(f"https://cloudflare-dns.com/dns-query?name={host}&type=A",
                                     headers={"accept": "application/dns-json"})
        import json
        return bool(json.loads(urllib.request.urlopen(req, timeout=5).read()).get("Answer"))
    except Exception:
        return None


def start_cloudflare_tunnel(port):
    import stat
    found = {"url": None, "registered": False, "ready": False}
    binary = "/tmp/cloudflared"
    try:
        if not os.path.exists(binary):
            urllib.request.urlretrieve(
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", binary)
            os.chmod(binary, os.stat(binary).st_mode | stat.S_IEXEC)
        tunnel = subprocess.Popen([binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate",
                                   "--protocol", "http2"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        import atexit
        atexit.register(tunnel.kill)
    except Exception as e:
        print(f"⚠️ Cloudflare tunnel unavailable: {e}")
        return found

    def _watch():
        for line in tunnel.stdout:  # keep draining so cloudflared never blocks on a full pipe
            match = re.search(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com", line)  # skip cloudflared's API host
            if match and not found["url"]:
                found["url"] = match.group(0)
            if "Registered tunnel connection" in line:
                found["registered"] = True

    def _wait_ready():
        deadline, registered_at = time.time() + 300, None
        while time.time() < deadline:
            if found["registered"] and registered_at is None:
                registered_at = time.time()
            published = _dns_published(found["url"].split("//", 1)[1]) if found["url"] and registered_at else False
            if published or (published is None and time.time() - registered_at > 15):
                found["ready"] = True
                print(f"\n✅ Cloudflare link is ready: {found['url']}\n")
                sys.stdout.flush()
                return
            time.sleep(3)
        print("⚠️ The Cloudflare link did not come up within 5 minutes; use the Gradio share link.")

    threading.Thread(target=_watch, daemon=True).start()
    threading.Thread(target=_wait_ready, daemon=True).start()
    return found



print("🌐 Starting the public links (Cloudflare + Gradio share)...")
sys.stdout.flush()
cf = start_cloudflare_tunnel(SERVER_PORT)
demo.queue(max_size=20)
demo.launch(server_name="127.0.0.1", server_port=SERVER_PORT, share=True, inline=False, show_error=True,
            prevent_thread_lock=True, quiet=True, ssr_mode=False, max_threads=8, allowed_paths=[OUTPUT_DIR])
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute


def _progress_route():
    end = PROGRESS["finished"] or time.time()
    elapsed = end - PROGRESS["started"] if PROGRESS["started"] else 0.0
    return JSONResponse({**PROGRESS, "elapsed": elapsed}, headers={"Cache-Control": "no-store"})


def _video_route():
    """The newest video as a plain file (plays in the browser, with a download option): no Gradio involved."""
    videos = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")), key=os.path.getmtime)
    if not videos:
        return JSONResponse({"error": "no video yet"}, status_code=404)
    name = os.path.basename(videos[-1])
    return FileResponse(videos[-1], media_type="video/mp4", headers={"Content-Disposition": f'inline; filename="{name}"',
                                                                      "Cache-Control": "no-store"})


for _path, _route in (("/aiq_progress", _progress_route), ("/aiq_video", _video_route)):
    demo.app.router.routes.insert(0, APIRoute(_path, _route, methods=["GET"]))  # ahead of Gradio's routes
for _ in range(60):  # give the tunnel up to ~30 s to come up before printing the links
    if cf["ready"]:
        break
    time.sleep(0.5)

print("\n" + "=" * 70)
print("🎬 MiniMax H3 Regenerate 2K is LIVE!")
print("=" * 70)
if cf["ready"]:
    print(f"🌐 Cloudflare tunnel: {cf['url']}")
else:
    print(f"🌐 Cloudflare tunnel: ⏳ warming up{' (' + cf['url'] + ')' if cf['url'] else ''} - wait for the "
          f"'✅ Cloudflare link is ready' line before opening it; opening it too early can keep it unreachable for minutes")
print(f"🔗 Gradio share:      {getattr(demo, 'share_url', None) or '(unavailable, use the Cloudflare link)'}")
print(f"💾 Videos are saved to {OUTPUT_DIR} (Kaggle Output panel); previews stay upscalable in {PREVIEW_DIR}")
print(f"🎞️ Latest video, without the UI: <link>/aiq_video (e.g. {cf['url'] or 'https://...trycloudflare.com'}/aiq_video)")
print("=" * 70 + "\n")
sys.stdout.flush()
while True:  # keep the server process alive; generation logs stream to the notebook cell
    time.sleep(3600)
'''

with open("run_h3_2k.py", "w", encoding="utf-8") as f:
    f.write(RUN_SCRIPT_CODE)
print("✅ run_h3_2k.py written.")

import json
if os.path.exists("h3_2k_settings.json"):  # Step 1's choices, for a Step 2 re-run after a kernel restart
    with open("h3_2k_settings.json") as f:
        for key, value in json.load(f).items():
            os.environ.setdefault(key, value)

IGNORE = ("absl::InitializeLog", "Unable to register cu", "computation placer already registered", "MessageFactory",
          "GetPrototype", "Attempting to register factory", "UserWarning", "warnings.warn", "FutureWarning",
          # library chatter that says nothing about this notebook's progress
          "Flax classes are deprecated", "[GGUF]", "Memory Management for the GPU Poor", "Pinning data of",
          "The whole model was pinned", "Hooked to model", "Async loading plan", "Loading MiniMax H3 Video VAE",
          "Running on public URL", "Running on local URL", "To create a public link", "[INT8] Backend")
env = dict(os.environ, PYTHONWARNINGS="ignore", TQDM_DISABLE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

print("🚀 Starting the MiniMax H3 Regenerate 2K engine...\n")
import psutil

def stop_old_engine():
    # A previous run's engine (or one left behind by a kernel restart) still holds both GPUs and ~24 GB of RAM.
    stale = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            args, name = p.info["cmdline"] or [], (p.info["name"] or "").lower()
            engine = "python" in name and any(os.path.basename(a) == "run_h3_2k.py" for a in args[1:3])
            if p.pid != os.getpid() and (engine or name == "cloudflared" or name.startswith("frpc")):
                p.kill()
                stale.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.wait_procs(stale, timeout=20)
    if stale:
        print(f"🧹 Stopped {len(stale)} leftover engine process(es) from an earlier run: GPU memory and RAM are free again")


stop_old_engine()

# Step 1's downloads leave freed buffers in this kernel process: hand them back to the OS before the engine starts.
import ctypes
import gc
gc.collect()
try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception:
    pass
_kernel_gb = psutil.Process().memory_info().rss / 1024**3
_ram = psutil.virtual_memory()
print(f"🧠 RAM before the engine starts: {(_ram.total - _ram.available) / 1024**3:.1f} GB in use "
      f"(this notebook kernel: {_kernel_gb:.1f} GB)")
if _kernel_gb > 2.0:
    print("   ⚠️ The kernel itself holds a lot of RAM. If the engine runs out of memory: Run → Restart, then run Step 2 only.")


def _die_with_kernel():
    # Linux: the engine gets SIGKILL when this notebook kernel dies or restarts (no orphan holding the GPUs).
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass


proc = subprocess.Popen([sys.executable, "-u", "run_h3_2k.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                        preexec_fn=_die_with_kernel)
import threading
_peak = {"ram": 0.0, "engine": 0.0}


def _watch_ram():  # peak RAM while the engine runs, reported if it dies
    engine = psutil.Process(proc.pid)
    while proc.poll() is None:
        try:
            vm = psutil.virtual_memory()
            _peak["ram"] = max(_peak["ram"], (vm.total - vm.available) / 1024**3)
            _peak["engine"] = max(_peak["engine"], engine.memory_info().rss / 1024**3)
        except Exception:
            pass
        time.sleep(0.5)


import time
threading.Thread(target=_watch_ram, daemon=True).start()
# Read raw bytes so carriage returns survive: the engine redraws its live progress line in place, and a text-mode
# pipe would turn every redraw into a new line.
decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
pending = ""
try:
    while True:
        chunk = os.read(proc.stdout.fileno(), 65536)
        if not chunk:
            break
        pending += decoder.decode(chunk)
        parts = re.split(r"(?<=[\r\n])", pending)
        pending = parts.pop()  # text after the last line break / carriage return
        out = "".join(part for part in parts if not any(pattern in part for pattern in IGNORE))
        if pending and not any(pattern in pending for pattern in IGNORE):
            out, pending = out + pending, ""  # show the live progress line right away
        if out:
            sys.stdout.write(out)
            sys.stdout.flush()
    sys.stdout.write(pending)
    rc = proc.wait()
    if rc != 0:
        import signal
        how = f"killed by {signal.Signals(-rc).name}" if rc < 0 else f"exit code {rc}"
        hint = (" - the system killed it, which almost always means it ran out of RAM" if rc == -9 else
                " - a crash inside a native library (see the traceback above)" if rc in (-11, -6) else "")
        print(f"\n❌ The engine stopped ({how}){hint}.\n   Peak RAM while it ran: {_peak['ram']:.1f} GB in use of "
              f"{psutil.virtual_memory().total / 1024**3:.1f} GB (engine process {_peak['engine']:.1f} GB).")
except KeyboardInterrupt:
    proc.kill()  # frees the GPUs and RAM right away
    proc.wait()
    print("🛑 Engine stopped. Re-run this cell to start it again.")