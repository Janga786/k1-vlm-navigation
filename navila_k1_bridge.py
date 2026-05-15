#!/usr/bin/env python3
"""NaVILA -> Booster K1 navigation bridge.

Pipeline:
    rolling 8-frame buffer -> NaVILA -> language action
    -> regex parse to (vx, vy, vyaw)
    -> dry-run print or B1LocoClient.Move on the K1 walker

NaVILA emits mid-level instructions like:
    "move forward 75 cm"
    "turn left 30 degrees"
    "turn right 15 degrees"
    "stop"

We hold each commanded velocity for ACTION_DURATION seconds. Velocity is
distance / duration (or angle / duration for turns), clipped to safety caps.

For initial smoke testing we re-feed a single static image into all 8
frame slots. Replace `get_image()` with a real camera grabber when the
robot is in hand.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import deque
from pathlib import Path

# torch + llava are lazy-imported inside load_navila() / run_inference() so
# the cheap helpers in this module (parse_action, build_prompt) stay usable
# from environments that don't have llava installed (e.g. unit tests).
from PIL import Image

DEFAULT_CKPT = Path.home() / "Projects/booster/NaVILA/checkpoints/navila-llama3-8b-8f"
NUM_FRAMES = 8

# Paper-spec command velocities (NaVILA, RSS 2025, Section II-B):
# VLM outputs cast to fixed speeds; per-action duration is what varies.
# 0.5 m/s for forward/backward, π/6 rad/s for turns.
import math as _math
FORWARD_SPEED = 0.5             # m/s
TURN_SPEED = _math.pi / 6.0     # rad/s

# Fallback duration when an action can't be parsed or carries no distance —
# kept for backwards compat with callers that still pass this through.
ACTION_DURATION = 1.5
SEND_HZ = 20.0


# ---- parsing ---------------------------------------------------------------

# Match phrases like "move forward 75 cm" / "move forward 0.5 m".
_FORWARD = re.compile(
    r"\b(?:move|walk|go|step)\s+(?:forward|ahead)\s+(?P<n>\d+(?:\.\d+)?)\s*(?P<u>cm|m|meter|meters|metre|metres)\b",
    re.I,
)
_BACKWARD = re.compile(
    r"\b(?:move|walk|go|step)\s+(?:back|backward|backwards)\s+(?P<n>\d+(?:\.\d+)?)\s*(?P<u>cm|m|meter|meters|metre|metres)\b",
    re.I,
)
_TURN = re.compile(
    r"\bturn\s+(?P<dir>left|right)\s+(?P<n>\d+(?:\.\d+)?)\s*(?P<u>deg|degree|degrees|rad|radian|radians)\b",
    re.I,
)
_STOP = re.compile(r"\b(?:stop|halt|done|complete[d]?)\b", re.I)


def _len_to_meters(n: float, unit: str) -> float:
    u = unit.lower()
    return n / 100.0 if u == "cm" else n


def _angle_to_rad(n: float, unit: str) -> float:
    import math
    u = unit.lower()
    return math.radians(n) if u.startswith("deg") else n


def parse_action(text: str, duration: float = ACTION_DURATION
                 ) -> tuple[float, float, float, float, str]:
    """Map a NaVILA text action → fixed-speed command + duration.

    Per NaVILA paper §II-B, the VLM's discrete output {forward, turn left,
    turn right, stop} is cast to **fixed command velocities** (0.5 m/s,
    ±π/6 rad/s, 0) and held for the time that matches the requested
    distance / angle. Speed is constant; duration varies.

    Returns ``(vx, vy, vyaw, duration_s, label)``. The ``duration`` arg is
    retained as a fallback for unparsed actions and is otherwise ignored.
    """
    if _STOP.search(text):
        return 0.0, 0.0, 0.0, 0.0, "stop"
    if (m := _FORWARD.search(text)):
        d = _len_to_meters(float(m["n"]), m["u"])
        dur = abs(d) / FORWARD_SPEED if FORWARD_SPEED > 0 else duration
        return FORWARD_SPEED, 0.0, 0.0, dur, f"forward {d:.2f}m"
    if (m := _BACKWARD.search(text)):
        d = _len_to_meters(float(m["n"]), m["u"])
        dur = abs(d) / FORWARD_SPEED if FORWARD_SPEED > 0 else duration
        return -FORWARD_SPEED, 0.0, 0.0, dur, f"backward {d:.2f}m"
    if (m := _TURN.search(text)):
        a = _angle_to_rad(float(m["n"]), m["u"])
        sign = 1.0 if m["dir"].lower() == "left" else -1.0
        dur = abs(a) / TURN_SPEED if TURN_SPEED > 0 else duration
        return 0.0, 0.0, sign * TURN_SPEED, dur, f"turn {m['dir']} {a:.2f}rad"
    return 0.0, 0.0, 0.0, 0.0, "unparsed -> stop"


# ---- VLM -------------------------------------------------------------------

def build_prompt(instruction: str, num_frames: int = NUM_FRAMES) -> str:
    image_token = "<image>\n"
    history_tokens = image_token * (num_frames - 1)
    return (
        f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
        f"of historical observations {history_tokens}, and current observation <image>\n. "
        f'Your assigned task is: "{instruction}" '
        f"Analyze this series of images to decide your next action, which could be turning left "
        f"or right by a specific degree, moving forward a certain distance, or stop if the task is completed."
    )


def load_navila(model_path: Path):
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    disable_torch_init()
    print(f"Loading NaVILA from {model_path} ...")
    model_name = get_model_name_from_path(str(model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(model_path), model_name, model_base=None,
        attn_implementation="sdpa",
    )
    print("NaVILA ready.")
    return tokenizer, model, image_processor


def run_inference(tokenizer, model, image_processor, frames: list[Image.Image],
                  instruction: str, max_new_tokens: int = 256) -> str:
    import torch
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria, process_images, tokenizer_image_token,
    )
    images_tensor = process_images(frames, image_processor, model.config).to(
        model.device, dtype=torch.float16)
    qs = build_prompt(instruction, len(frames))
    conv = conv_templates["llama_3"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor.half().cuda(),
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping],
            pad_token_id=tokenizer.eos_token_id,
        )

    out = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if out.endswith(stop_str):
        out = out[: -len(stop_str)].strip()
    return out


# ---- camera + sender stubs -------------------------------------------------

def make_image_source(image_path: Path):
    """Return a callable that yields a PIL.Image each tick.

    Stand-in for a real camera grabber. Replace when the robot is wired up.
    """
    if not image_path.exists():
        raise SystemExit(f"image not found: {image_path}")
    return lambda: Image.open(image_path).convert("RGB")


def make_sender(live: bool, net: str):
    """Return (send_fn(vx,vy,vyaw), cleanup_fn)."""
    if not live:
        def dry(vx, vy, vyaw):
            print(f"   [dry] Move({vx:+.3f}, {vy:+.3f}, {vyaw:+.3f})")
        return dry, lambda: None

    try:
        from booster_robotics_sdk_python import (
            B1LocoClient, ChannelFactory, RobotMode,
        )
    except ImportError as e:
        sys.exit(
            f"booster_robotics_sdk_python is not installed in this env: {e}\n"
            f"Build the binding for the navila env (same procedure as for vla)\n"
            f"or run --dry-run.")

    print(f"Initializing channel on {net} ...")
    ChannelFactory.Instance().Init(0, net)
    client = B1LocoClient()
    client.Init()
    print("Switching to kWalking ...")
    client.ChangeMode(RobotMode.kWalking)
    time.sleep(0.5)

    def cleanup():
        try:
            client.Move(0.0, 0.0, 0.0)
            time.sleep(0.2)
            client.ChangeMode(RobotMode.kDamping)
            print("Switched to kDamping.")
        except Exception as e:
            print(f"Cleanup failed: {e}", file=sys.stderr)

    return client.Move, cleanup


# ---- main ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instruction", required=True,
                    help='navigation instruction, e.g. "walk to the chair"')
    ap.add_argument("--model-path", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--image",
                    default=str(Path.home() / "Projects/k1_research/experiments/vla/test_image.jpg"),
                    help="image path (camera stand-in)")
    ap.add_argument("--steps", type=int, default=20,
                    help="number of NaVILA inference iterations")
    ap.add_argument("--action-duration", type=float, default=ACTION_DURATION,
                    help="seconds to hold each NaVILA action before re-asking")
    ap.add_argument("--send-hz", type=float, default=SEND_HZ,
                    help="Move() streaming rate while holding an action")
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="print commands instead of sending them (default)")
    ap.add_argument("--live", action="store_true",
                    help="actually send via SDK (overrides --dry-run)")
    ap.add_argument("--net", default="127.0.0.1",
                    help="ChannelFactory network interface / robot IP")
    args = ap.parse_args()

    live = args.live
    get_image = make_image_source(Path(args.image))
    tokenizer, model, image_processor = load_navila(args.model_path)
    send, cleanup = make_sender(live, args.net)

    # Rolling 8-frame buffer; bootstrap by repeating the first frame.
    frames: deque[Image.Image] = deque(maxlen=NUM_FRAMES)
    first = get_image()
    for _ in range(NUM_FRAMES):
        frames.append(first)

    period = 1.0 / args.send_hz
    print(f"\nInstruction: {args.instruction!r}")
    print(f"Mode: {'LIVE' if live else 'DRY'}   "
          f"action_duration={args.action_duration}s   send_hz={args.send_hz}\n")

    try:
        for step in range(args.steps):
            frames.append(get_image())
            t0 = time.perf_counter()
            text = run_inference(tokenizer, model, image_processor,
                                 list(frames), args.instruction)
            inf_dt = time.perf_counter() - t0
            vx, vy, vyaw, duration, label = parse_action(text, args.action_duration)
            print(f"step {step:3d}  inf={inf_dt*1000:6.0f} ms  "
                  f"raw={text!r}\n            "
                  f"-> {label}  vx={vx:+.3f} vy={vy:+.3f} vyaw={vyaw:+.3f} "
                  f"hold={duration:.2f}s")

            if label == "stop":
                send(0.0, 0.0, 0.0)
                print("NaVILA emitted stop. Exiting loop.")
                break

            # Stream the chosen velocity for the per-action duration
            # (NaVILA paper §II-B: fixed speed, distance/speed seconds).
            t_end = time.perf_counter() + duration
            next_tick = time.perf_counter()
            while time.perf_counter() < t_end:
                send(vx, vy, vyaw)
                next_tick += period
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.perf_counter()
    finally:
        send(0.0, 0.0, 0.0)
        cleanup()


if __name__ == "__main__":
    main()
