#!/usr/bin/env python3
"""Deploy pre-flight: load NaVILA at fp16 on the (now-free) GPU and run ONE real
inference through the deploy's OWN code path — no robot, no DDS needed.

Proves the VLM half of the deploy pipeline end-to-end:
  load fp16 -> real K1 camera frame -> prep_frame (undistort+stretch) -> NaVILA
  generate -> parse_action -> (vx,vy,vyaw,label).

Run (navila env):
  NAVILA_VLM_TRANSFORM=stretch python preflight_vlm.py [sample.jpg]
Exit 0 = VLM deploy path GREEN.
"""
import os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("NAVILA_VLM_TRANSFORM", "stretch")  # matches the 18.3% benchmark run

import numpy as np
from PIL import Image
import torch

# deploy modules (same imports the real robot uses)
import navila_k1_core as core
from navila_k1_bridge import NUM_FRAMES, parse_action, ACTION_DURATION

CKPT = Path.home() / "Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f"
SAMPLE = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path.home() / "receipt_validator/camera_frames/live.jpg"
INSTRUCTION = "Walk forward through the room and stop at the doorway."


def gpu_mem():
    try:
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e9, total / 1e9
    except Exception:
        return -1, -1


def main():
    print(f"[preflight] transform={os.environ['NAVILA_VLM_TRANSFORM']} "
          f"undistort={core.UNDISTORT}  ckpt={CKPT.name}")
    assert CKPT.exists(), f"checkpoint missing: {CKPT}"
    assert SAMPLE.exists(), f"sample frame missing: {SAMPLE}"

    # --- load exactly as deploy does (fp16, no 8bit) ---
    vila = os.environ.get("NAVILA_VILA_REPO",
                          str(Path.home() / "Projects/k1_research/experiments/navila"))
    if Path(vila).exists() and vila not in sys.path:
        sys.path.insert(0, vila)
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.utils import disable_torch_init
    disable_torch_init()
    t0 = time.perf_counter()
    name = get_model_name_from_path(str(CKPT))
    tok, model, image_processor, _ = load_pretrained_model(
        str(CKPT), model_base=None, model_name=name, attn_implementation="sdpa")
    load_s = time.perf_counter() - t0
    used, total = gpu_mem()
    dtype = next(model.parameters()).dtype
    print(f"[preflight] LOADED in {load_s:.0f}s  dtype={dtype}  "
          f"GPU {used:.1f}/{total:.1f} GB used")
    assert dtype in (torch.float16, torch.bfloat16), f"expected fp16, got {dtype}"

    # --- real frame -> deploy prep -> 8-frame buffer ---
    raw = Image.open(SAMPLE).convert("RGB")
    print(f"[preflight] sample {SAMPLE.name} {raw.size}")
    prepped = core.prep_frame(raw)
    print(f"[preflight] after prep_frame: {prepped.size} (square={prepped.size[0]==prepped.size[1]})")
    frames = [prepped] * NUM_FRAMES

    # --- inference (the deploy's module-level fn) ---
    t1 = time.perf_counter()
    raw_out = core._navila_inference(tok, model, image_processor, frames, INSTRUCTION)
    inf_ms = (time.perf_counter() - t1) * 1000.0
    vx, vy, vyaw, label = parse_action(raw_out, ACTION_DURATION)

    print("\n================ PRE-FLIGHT RESULT ================")
    print(f"instruction : {INSTRUCTION!r}")
    print(f"raw NaVILA   : {raw_out!r}")
    print(f"parsed       : label={label}  vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f}")
    print(f"inference    : {inf_ms:.0f} ms   (load {load_s:.0f}s, GPU {used:.1f}/{total:.1f} GB)")
    ok = bool(raw_out) and (label == "stop" or label.startswith(("forward", "backward", "turn")))
    note = "GREEN — VLM deploy path works at fp16" if ok else \
        f"CHECK — raw output did not parse to a motion/stop (label={label!r})"
    print(f"VERDICT      : {note}")
    print("===================================================")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
