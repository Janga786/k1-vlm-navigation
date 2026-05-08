#!/usr/bin/env python3
"""Closed-loop NaVILA → walking K1 → MuJoCo simulation.

This is the full NaVILA-on-K1 demo: NaVILA decides velocity commands from
the head camera, the trained K1 velocity-tracking policy turns them into joint
torques, and the K1 *physically walks* (legs swing, feet step) — no kinematic
sliding of the floating base.

Architecture (one process in the `navila` conda env)
====================================================
- main thread:
    physics + walking policy at 50 Hz (decimation 10 × 200 Hz physics)
    head + scene cameras rendered at ~30 Hz, written to two MP4s
    head frames pushed into an 8-frame ring buffer for the VLM
- VLM thread:
    runs NaVILA generate() in a tight loop on the latest buffer snapshot
    parses "move forward 75 cm" / "turn left 30 deg" / "stop" → (vx, vy, vyaw)
    publishes the new command under a lock; main thread reads it each step
    when NaVILA emits "stop" the thread sets a stop event and the main loop
    drains for ~2 s so the policy decelerates the K1 cleanly.

The walking policy (`booster_deploy/tasks/locomotion/k1_velocity.py`) builds
the 235-dim observation (cmd | gait | gravity | ang_vel | joint_pos_rel |
joint_vel | last_action) × 5-frame term-major history, runs the 235→12 MLP,
and scatters the leg targets into a 22-DoF default pose. We just keep the
shared command tensor in `controller.vel_command` updated.

`MUJOCO_GL=egl` is set before any mujoco imports because the GLFW backend
clashes with torch's CUDA init in this process.

Usage
-----
    /home/janga/miniconda3/envs/navila/bin/python navila_k1_walking_loop.py \\
        --instruction "navigate to the red box" \\
        --steps 30 --save-video ./out
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")  # before mujoco imports

import argparse
import math
import re
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --- path injection so we can import everything from one process -----------
_K1RES = Path.home() / "Projects" / "k1_research"
_NAVILA_REPO = _K1RES / "booster" / "NaVILA"
_BOOSTER_ASSETS_SRC = _K1RES / "booster" / "booster_assets" / "src"
_BOOSTER_DEPLOY = _K1RES / "booster" / "booster_deploy"
for p in (_NAVILA_REPO, _BOOSTER_ASSETS_SRC, _BOOSTER_DEPLOY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# --- third-party imports ----------------------------------------------------
import cv2
import mujoco
import numpy as np
import torch
from PIL import Image

# --- VLM imports ------------------------------------------------------------
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria, get_model_name_from_path,
    process_images, tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

# --- bridge (action parser, prompt builder) ---------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from navila_k1_bridge import (  # type: ignore  # noqa: E402
    NUM_FRAMES, build_prompt, parse_action, ACTION_DURATION,
)

# --- booster_deploy walking policy ------------------------------------------
import pkgutil
import tasks as _deploy_tasks
for _m in pkgutil.walk_packages(_deploy_tasks.__path__, prefix="tasks."):
    __import__(_m.name)
from booster_deploy.controllers.mujoco_controller import MujocoController
from booster_deploy.utils.registry import get_task

K1_XML = (_K1RES / "booster" / "booster_assets" / "robots" / "K1" /
          "K1_22dof.xml")
DEFAULT_CKPT = (_NAVILA_REPO / "checkpoints" / "navila-llama3-8b-8f")


# ============================================================================
# Scene XML: K1 + head_cam + scene_cam + target + distractors
# ============================================================================

def build_scene_xml(robot_xml_path: Path,
                    target_pos: tuple[float, float, float],
                    distractors: list[tuple[str, tuple[float, float, float], str, str]]
                    ) -> str:
    """Augment K1_22dof.xml with cameras, target, and distractors.

    Returns a complete MJCF as a string (with absolute meshdir so the file can
    be loaded from anywhere).
    """
    tree = ET.parse(robot_xml_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((robot_xml_path.parent / "meshes").resolve()))

    # Big offscreen framebuffer so head + scene renders both fit. Also bump
    # shadow map size and add a headlight so the EGL backend produces clean
    # frames without the speckled cubemap-reflection artifacts you get from
    # the MuJoCo defaults at small shadow resolutions.
    visual = root.find("visual") or ET.SubElement(root, "visual")
    global_el = visual.find("global") or ET.SubElement(visual, "global")
    global_el.set("offwidth", "1280")
    global_el.set("offheight", "960")
    quality = visual.find("quality") or ET.SubElement(visual, "quality")
    quality.set("shadowsize", "4096")
    quality.set("offsamples", "4")        # 4× MSAA on offscreen FB
    headlight = visual.find("headlight") or ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.4 0.4 0.4")
    headlight.set("diffuse", "0.5 0.5 0.5")
    headlight.set("specular", "0.0 0.0 0.0")
    rgba = visual.find("rgba") or ET.SubElement(visual, "rgba")
    rgba.set("haze", "0.15 0.25 0.35 1")  # softens the horizon

    # Kill matplane reflectance — that's the source of the bright white
    # blotches (cubemap reflection of sky/objects sampled across the floor).
    asset = root.find("asset")
    if asset is not None:
        for mat in asset.iter("material"):
            if mat.get("name") in {"matplane", "floor", "groundplane"}:
                mat.set("reflectance", "0")
                mat.set("shininess", "0")
                mat.set("specular", "0")

    # --- head_cam mounted on Head_2 (matches sliding demo placement) --------
    head2 = next((b for b in root.iter("body") if b.get("name") == "Head_2"), None)
    if head2 is None:
        raise RuntimeError("Head_2 body not found in K1 MJCF")
    cam = ET.SubElement(head2, "camera")
    cam.set("name", "head_cam")
    cam.set("pos", "0.07 0 0.08")        # ~ZED bezel front-and-up of Head_2
    cam.set("xyaxes", "0 -1 0 0 0 1")    # face body +X, image up = body +Z
    cam.set("fovy", "70")

    worldbody = root.find("worldbody")

    # --- third-person tracking camera ---------------------------------------
    scene_cam = ET.SubElement(worldbody, "camera")
    scene_cam.set("name", "scene_cam")
    scene_cam.set("mode", "targetbodycom")
    scene_cam.set("target", "Trunk")
    scene_cam.set("pos", "1.5 4.0 2.0")
    scene_cam.set("fovy", "60")

    # The K1 MJCF already ships with a `<geom name="ground">` plane backed by
    # the `matplane` material. Adding a second coplanar floor causes severe
    # z-fighting (the bright blotches we were seeing). Just retune the
    # existing plane's friction + condim to match the training contact model
    # (Isaac Lab static_friction=1.0, dynamic_friction=1.0, restitution=0).
    existing_floor = None
    for g in worldbody.iter("geom"):
        if g.get("type") == "plane" or g.get("name") in {"ground", "floor"}:
            existing_floor = g
            break
    if existing_floor is not None:
        existing_floor.set("friction", "1.0 0.005 0.0001")
        existing_floor.set("condim", "3")
        existing_floor.set("size", "20 20 0.1")
    else:
        floor = ET.SubElement(worldbody, "geom")
        floor.set("name", "ground")
        floor.set("type", "plane")
        floor.set("size", "20 20 0.1")
        floor.set("pos", "0 0 0")
        floor.set("rgba", "0.55 0.6 0.65 1")
        floor.set("friction", "1.0 0.005 0.0001")
        floor.set("condim", "3")

    # --- target (red box) ---------------------------------------------------
    target = ET.SubElement(worldbody, "body")
    target.set("name", "navigation_target")
    target.set("pos", f"{target_pos[0]} {target_pos[1]} {target_pos[2]}")
    g = ET.SubElement(target, "geom")
    g.set("type", "box")
    g.set("size", "0.20 0.20 0.30")
    g.set("rgba", "0.92 0.10 0.10 1")
    g.set("contype", "0")          # no collisions with the K1
    g.set("conaffinity", "0")

    # --- distractors --------------------------------------------------------
    for name, pos, rgba, sz in distractors:
        b = ET.SubElement(worldbody, "body")
        b.set("name", name)
        b.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
        gg = ET.SubElement(b, "geom")
        gg.set("type", "box")
        gg.set("size", sz)
        gg.set("rgba", rgba)
        gg.set("contype", "0")
        gg.set("conaffinity", "0")

    return ET.tostring(root, encoding="unicode")


# ============================================================================
# Multi-step instruction planner
# ============================================================================


@dataclass
class SubStep:
    """One atomic instruction handed to NaVILA, plus how we know it is done.

    Termination priority (first match wins):
      1. ``proximity_target`` reached within ``proximity_threshold`` (m, xy).
      2. ``yaw_delta_target`` reached (signed radians, accumulated since the
         start of this sub-step).
      3. NaVILA emits "stop".
      4. ``time_limit`` seconds elapsed.
    """
    instruction: str
    time_limit: float = 30.0
    proximity_target: Optional[tuple[float, float, float]] = None
    proximity_threshold: float = 0.6
    yaw_delta_target: Optional[float] = None  # signed radians (left=+)


# Sub-step splitter: pipe, semicolon, or "then" (with optional comma).
_SPLIT = re.compile(r"\s*\|\s*|\s*;\s*|,?\s+then\s+", re.I)
# Turn parser: matches the same phrases NaVILA emits + plain user input.
_TURN_PAT = re.compile(
    r"\bturn\s+(?P<dir>left|right)\s+(?P<n>\d+(?:\.\d+)?)"
    r"\s*(?P<u>deg|degree|degrees|rad|radian|radians)\b",
    re.I,
)


def parse_substeps(text: str,
                   scene_targets: dict[str, tuple[float, float, float]],
                   default_time: float,
                   proximity_threshold: float,
                   ) -> list[SubStep]:
    """Decompose a multi-step instruction into ``SubStep``s.

    Each chunk is the instruction we feed NaVILA verbatim. We additionally
    inspect the text for cues that let us terminate the sub-step from the
    outside, which is more reliable than waiting for NaVILA's "stop":

    - ``"turn (left|right) N deg"``  → yaw-delta termination
    - mentions of a known scene target (e.g. "red box") → proximity termination
    """
    chunks = [c.strip() for c in _SPLIT.split(text) if c.strip()]
    if not chunks:
        chunks = [text.strip()]

    steps: list[SubStep] = []
    for c in chunks:
        ss = SubStep(instruction=c, time_limit=default_time,
                     proximity_threshold=proximity_threshold)

        m = _TURN_PAT.search(c)
        if m:
            sign = 1.0 if m.group("dir").lower() == "left" else -1.0
            n = float(m.group("n"))
            unit = m.group("u").lower()
            angle = math.radians(n) if unit.startswith("deg") else n
            ss.yaw_delta_target = sign * angle

        # Match the LAST scene target named in the instruction so a phrase
        # like "walk past the blue box to the red box" picks "red box".
        last_match_at = -1
        for name, pos in scene_targets.items():
            m2 = list(re.finditer(rf"\b{re.escape(name)}\b", c, re.I))
            if m2 and m2[-1].start() > last_match_at:
                last_match_at = m2[-1].start()
                ss.proximity_target = pos

        # If both yaw and proximity were detected (e.g. "turn right 90 deg
        # toward the red box"), prefer yaw — it's the explicit primitive.
        if ss.yaw_delta_target is not None and ss.proximity_target is not None:
            ss.proximity_target = None

        steps.append(ss)
    return steps


def describe_substep(i: int, n: int, ss: SubStep) -> str:
    bits = [f"step {i + 1}/{n}: {ss.instruction!r}"]
    if ss.yaw_delta_target is not None:
        bits.append(f"yaw_target={math.degrees(ss.yaw_delta_target):+.0f}°")
    if ss.proximity_target is not None:
        bits.append(f"proximity={ss.proximity_target[:2]} (<{ss.proximity_threshold}m)")
    bits.append(f"time_limit={ss.time_limit:.0f}s")
    return "  ".join(bits)


def wrap_pi(a: float) -> float:
    return float(((a + math.pi) % (2.0 * math.pi)) - math.pi)


# ============================================================================
# Walking + camera controller
# ============================================================================

class WalkingSceneController(MujocoController):
    """MujocoController, but loads the model from an XML *string* so we can
    splice in cameras, target, and distractors at runtime.
    """

    def __init__(self, cfg, scene_xml: str):
        # Replicate parent __init__ but use from_xml_string.
        from booster_deploy.controllers.base_controller import BaseController
        BaseController.__init__(self, cfg)

        self.mj_model = mujoco.MjModel.from_xml_string(scene_xml)
        self.mj_model.opt.timestep = self.cfg.mujoco.physics_dt
        self.decimation = self.cfg.mujoco.decimation
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # Initial pose: floating base + 22 joint defaults from the cfg.
        self.mj_data.qpos = np.concatenate([
            np.array(self.cfg.mujoco.init_pos, dtype=np.float32),
            np.array(self.cfg.mujoco.init_quat, dtype=np.float32),
            self.robot.default_joint_pos.numpy(),
        ])
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Stubs the parent uses for the (unused) reference-ghost rendering.
        self._ghost_mj_data = mujoco.MjData(self.mj_model)
        self._ghost_mj_data.qpos[:] = self.mj_data.qpos
        self._ghost_mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self._ghost_mj_data)
        self._ghost_rgba = np.array(
            self.cfg.mujoco.ghost_rgba, dtype=np.float32)
        self._ghost_scene_option = mujoco.MjvOption()
        self._reference_qpos = None

    # The parent's update_vel_command() polls stdin which we don't want here;
    # ctrl_step() calls it, so override with a no-op. Velocity is set by the
    # VLM thread.
    def update_vel_command(self) -> None:  # type: ignore[override]
        return None


# ============================================================================
# NaVILA loader and inference helpers
# ============================================================================

def load_navila(model_path: Path):
    disable_torch_init()
    print(f"[VLM] Loading NaVILA from {model_path} ...", flush=True)
    model_name = get_model_name_from_path(str(model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(model_path), model_name, model_base=None,
        attn_implementation="sdpa",
    )
    print("[VLM] NaVILA ready.", flush=True)
    return tokenizer, model, image_processor


@torch.inference_mode()
def run_inference(tokenizer, model, image_processor,
                  frames: list[Image.Image], instruction: str,
                  max_new_tokens: int = 64) -> str:
    images_tensor = process_images(
        frames, image_processor, model.config
    ).to(model.device, dtype=torch.float16)
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
    out_ids = model.generate(
        input_ids,
        images=images_tensor.half().cuda(),
        do_sample=False,
        temperature=0.0,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        stopping_criteria=[stopping],
        pad_token_id=tokenizer.eos_token_id,
    )
    out = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    if out.endswith(stop_str):
        out = out[: -len(stop_str)].strip()
    return out


# ============================================================================
# VLM background thread
# ============================================================================

class VLMRunner:
    """Continuous NaVILA inference on the latest head_cam buffer.

    The main thread pushes head frames; this thread keeps running NaVILA on
    the latest snapshot and publishes the parsed velocity command. When NaVILA
    emits "stop", `stop_event` is set so the main loop wraps up.
    """

    def __init__(
        self,
        model_path: Path,
        action_duration: float,
        vx_max: float,
        vy_max: float,
        vyaw_max: float,
    ):
        self.action_duration = action_duration
        self.vx_max, self.vy_max, self.vyaw_max = vx_max, vy_max, vyaw_max

        self.tokenizer, self.model, self.image_processor = load_navila(model_path)

        self._lock = threading.Lock()
        self._cmd = (0.0, 0.0, 0.0)
        self._label = "(waiting on first vlm result)"
        self._raw_text = ""
        self._inf_count = 0
        self._inf_ms = 0.0

        self._instruction_lock = threading.Lock()
        self._instruction = "(none)"

        self._frame_lock = threading.Lock()
        self._frame_buffer: deque[Image.Image] = deque(maxlen=NUM_FRAMES)

        self.stop_event = threading.Event()      # set when "stop" parsed
        self._abort = threading.Event()          # set by main to quit
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ API
    def push_frame(self, rgb: np.ndarray) -> None:
        img = Image.fromarray(rgb)
        with self._frame_lock:
            self._frame_buffer.append(img)

    def bootstrap_buffer(self, rgb: np.ndarray) -> None:
        """Fill the buffer with copies of `rgb` (used at startup so NaVILA
        always has 8 frames to look at)."""
        img = Image.fromarray(rgb)
        with self._frame_lock:
            for _ in range(NUM_FRAMES):
                self._frame_buffer.append(img)

    def get_command(self) -> tuple[float, float, float]:
        with self._lock:
            return self._cmd

    def set_instruction(self, text: str) -> None:
        """Switch the active instruction (called by main on sub-step advance)."""
        with self._instruction_lock:
            self._instruction = text
        # Force the published command back to zero so the policy doesn't
        # carry the previous sub-step's cmd into this one until the first
        # NaVILA result for the new instruction lands.
        with self._lock:
            self._cmd = (0.0, 0.0, 0.0)
            self._label = f"(switching to: {text})"

    def get_instruction(self) -> str:
        with self._instruction_lock:
            return self._instruction

    def clear_stop(self) -> None:
        """Reset the stop signal so the next sub-step starts fresh."""
        self.stop_event.clear()

    def status(self) -> dict:
        with self._lock:
            return {
                "label": self._label,
                "raw": self._raw_text,
                "vx": self._cmd[0],
                "vy": self._cmd[1],
                "vyaw": self._cmd[2],
                "inf_count": self._inf_count,
                "inf_ms": self._inf_ms,
            }

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="vlm",
                                         daemon=True)
        self._thread.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        self._abort.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------ loop
    def _snapshot(self) -> list[Image.Image]:
        with self._frame_lock:
            return list(self._frame_buffer)

    def _publish(self, vx: float, vy: float, vyaw: float, label: str,
                 raw: str, inf_ms: float) -> None:
        with self._lock:
            self._cmd = (vx, vy, vyaw)
            self._label = label
            self._raw_text = raw
            self._inf_count += 1
            self._inf_ms = inf_ms

    def _loop(self) -> None:
        # Wait until the main thread bootstraps the buffer AND sets the first
        # instruction. We don't want NaVILA running with "(none)".
        while not self._abort.is_set() and (
            len(self._snapshot()) == 0
            or self.get_instruction() == "(none)"
        ):
            time.sleep(0.05)

        while not self._abort.is_set():
            frames = self._snapshot()
            if len(frames) < NUM_FRAMES:
                # pad: repeat oldest frame to reach NUM_FRAMES
                frames = [frames[0]] * (NUM_FRAMES - len(frames)) + frames
            instruction = self.get_instruction()
            t0 = time.perf_counter()
            try:
                raw = run_inference(self.tokenizer, self.model,
                                    self.image_processor, frames,
                                    instruction)
            except Exception as e:
                print(f"[VLM] inference failed: {e!r}", flush=True)
                time.sleep(0.5)
                continue
            inf_ms = (time.perf_counter() - t0) * 1000.0

            vx, vy, vyaw, label = parse_action(raw, self.action_duration)
            # Re-clip to our (possibly higher) caps; navila_k1_bridge clips to
            # its conservative VX_MAX/VY_MAX/VYAW_MAX which we may want to
            # raise.
            vx = max(-self.vx_max, min(self.vx_max, vx))
            vy = max(-self.vy_max, min(self.vy_max, vy))
            vyaw = max(-self.vyaw_max, min(self.vyaw_max, vyaw))

            self._publish(vx, vy, vyaw, label, raw, inf_ms)
            print(f"[VLM #{self._inf_count:03d} {inf_ms:5.0f}ms] "
                  f"task={instruction!r}\n"
                  f"            raw={raw!r}\n"
                  f"            -> {label}  "
                  f"vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f}", flush=True)

            # Set the stop event but DO NOT terminate the thread — the main
            # loop may advance us to the next sub-step.
            if label == "stop":
                self.stop_event.set()


# ============================================================================
# Main loop
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instruction", required=True,
                     help='Navigation instruction. Use `|`, `;`, or `then` to '
                          'chain sub-steps. Each sub-step runs until a turn '
                          'angle is reached, the named target is in proximity, '
                          'NaVILA emits "stop", or --per-step-time elapses.\n'
                          'Examples:\n'
                          '  "navigate to the red box"\n'
                          '  "walk to the red box | turn right 90 deg | '
                          'walk forward"')
    ap.add_argument("--model-path", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--target", type=float, nargs=3, default=(3.0, 0.0, 0.30),
                     metavar=("X", "Y", "Z"),
                     help="World-frame position of the red target box.")
    ap.add_argument("--steps", type=int, default=200,
                     help="Hard cap on total NaVILA inferences across all "
                          "sub-steps.")
    ap.add_argument("--per-step-time", type=float, default=25.0,
                     help="Per-sub-step wall-clock time budget (seconds).")
    ap.add_argument("--proximity-threshold", type=float, default=1.0,
                     help="A sub-step that mentions a known target is "
                          "completed when the K1 base is within this many "
                          "metres of the target's xy position.")
    ap.add_argument("--heading-assist", action=argparse.BooleanOptionalAction,
                     default=True,
                     help="Overlay a P-controller on yaw that points the K1 "
                          "at the active proximity target. Cancels the slow "
                          "policy drift that NaVILA cannot compensate for.")
    ap.add_argument("--heading-kp", type=float, default=1.5,
                     help="Heading-assist proportional gain (rad/s per rad).")
    ap.add_argument("--turn-controller", action=argparse.BooleanOptionalAction,
                     default=True,
                     help="For pure-turn sub-steps ('turn left/right N deg'), "
                          "bypass NaVILA and drive yaw directly with a P "
                          "controller. NaVILA tends to emit 'move forward' "
                          "even when asked to turn.")
    ap.add_argument("--turn-kp", type=float, default=2.0,
                     help="Turn-controller P gain (rad/s per rad of yaw "
                          "remaining).")
    ap.add_argument("--turn-tolerance-deg", type=float, default=5.0,
                     help="Yaw-target sub-step terminates when the K1 is "
                          "within this many degrees of the target. The "
                          "discrete walking gait cannot reliably execute "
                          "tiny vyaw commands so we don't wait for a perfect "
                          "landing.")
    ap.add_argument("--turn-min-vyaw", type=float, default=0.30,
                     help="Floor on the magnitude of the turn-controller's "
                          "vyaw output (rad/s). Below ~0.25 the gait does "
                          "not actually rotate the base.")
    ap.add_argument("--closest-approach-margin", type=float, default=0.25,
                     help="Sub-step done if distance grows by more than this "
                          "many metres past its minimum (catches 'walked past "
                          "the target').")
    ap.add_argument("--closest-approach-min", type=float, default=1.5,
                     help="Closest-approach termination only triggers once "
                          "the K1 has gotten at least this close (m).")
    ap.add_argument("--max-sim-seconds", type=float, default=120.0,
                     help="Hard wall-clock cap on the entire simulation.")
    ap.add_argument("--action-duration", type=float, default=ACTION_DURATION,
                     help="Hold each VLM action for this many seconds when "
                          "translating distance/angle to velocity.")
    ap.add_argument("--render-hz", type=float, default=30.0)
    ap.add_argument("--head-w", type=int, default=640)
    ap.add_argument("--head-h", type=int, default=480)
    ap.add_argument("--scene-w", type=int, default=960)
    ap.add_argument("--scene-h", type=int, default=720)
    ap.add_argument("--vx-max", type=float, default=0.6)
    ap.add_argument("--vy-max", type=float, default=0.3)
    ap.add_argument("--vyaw-max", type=float, default=0.6)
    ap.add_argument("--save-video", type=Path, default=None,
                     help="Write scene_view.mp4 + head_view.mp4 to this dir.")
    ap.add_argument("--no-vlm", action="store_true",
                     help="Skip NaVILA; drive with constant vx instead "
                          "(useful to test the walking pipeline).")
    ap.add_argument("--debug-vx", type=float, default=0.4,
                     help="Used with --no-vlm.")
    args = ap.parse_args()

    # ------------------------------------------------------------------ scene
    print(f"[scene] red target = {tuple(args.target)}")
    blue_pos  = (2.0, -1.5, 0.25)
    green_pos = (1.5,  1.8, 0.25)
    distractors = [
        ("distractor_blue",  blue_pos,  "0.10 0.30 0.85 1", "0.18 0.18 0.25"),
        ("distractor_green", green_pos, "0.10 0.70 0.20 1", "0.18 0.18 0.25"),
    ]
    scene_xml = build_scene_xml(K1_XML, tuple(args.target), distractors)

    # Map every name a user might say to the world-pos it should aim for.
    scene_targets: dict[str, tuple[float, float, float]] = {
        "red box":   tuple(args.target),
        "red cube":  tuple(args.target),
        "red":       tuple(args.target),
        "blue box":  blue_pos,
        "blue cube": blue_pos,
        "blue":      blue_pos,
        "green box":  green_pos,
        "green cube": green_pos,
        "green":      green_pos,
    }

    # ----------------------------------------------------------------- plan
    substeps = parse_substeps(
        args.instruction, scene_targets,
        default_time=args.per_step_time,
        proximity_threshold=args.proximity_threshold,
    )
    print(f"[plan] {len(substeps)} sub-step(s):")
    for i, ss in enumerate(substeps):
        print("  " + describe_substep(i, len(substeps), ss))

    # The booster_deploy MujocoController loads from a path. We keep our own
    # subclass that takes the XML string directly, but we still need the cfg.
    # k1_velocity registers the K1 22-DoF mjcf path; we won't touch it because
    # WalkingSceneController bypasses that loading path.
    cfg = get_task("k1_velocity")

    # ----------------------------------------------------------------- writer
    if args.save_video is not None:
        args.save_video.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        scene_writer = cv2.VideoWriter(
            str(args.save_video / "scene_view.mp4"), fourcc, args.render_hz,
            (args.scene_w, args.scene_h))
        head_writer = cv2.VideoWriter(
            str(args.save_video / "head_view.mp4"), fourcc, args.render_hz,
            (args.head_w, args.head_h))
        if not scene_writer.isOpened() or not head_writer.isOpened():
            sys.exit("Failed to open video writer (try installing "
                     "opencv-python-headless or different fourcc).")
        print(f"[video] writing to {args.save_video}/")
    else:
        scene_writer = head_writer = None

    # --------------------------------------------------------- controller
    controller = WalkingSceneController(cfg, scene_xml)
    controller.update_state()
    controller.start()  # also resets the policy's gait phase / history

    head_renderer = mujoco.Renderer(controller.mj_model,
                                     height=args.head_h, width=args.head_w)
    scene_renderer = mujoco.Renderer(controller.mj_model,
                                      height=args.scene_h, width=args.scene_w)

    # --------------------------------------------------------- VLM
    if args.no_vlm:
        vlm = None
        const_cmd = (args.debug_vx, 0.0, 0.0)
        print(f"[main] --no-vlm: holding cmd = {const_cmd}")
    else:
        vlm = VLMRunner(
            args.model_path, args.action_duration,
            args.vx_max, args.vy_max, args.vyaw_max,
        )
        bootstrap_rgb = render_camera(head_renderer, controller.mj_data, "head_cam")
        vlm.bootstrap_buffer(bootstrap_rgb)
        vlm.set_instruction(substeps[0].instruction)
        vlm.start()

    # --------------------------------------------------------- run loop
    policy_dt = float(cfg.policy_dt)              # 0.02
    render_period = 1.0 / args.render_hz
    head_buffer_period = 0.4                       # one head frame every 0.4 s
    sim_t0 = time.perf_counter()
    next_render = time.perf_counter()
    next_head_push = time.perf_counter()
    next_print = time.perf_counter()

    drain_deadline: float | None = None  # set when all sub-steps done

    # Per-sub-step bookkeeping.
    step_idx = 0
    step_start_time = time.perf_counter()
    step_start_yaw = yaw_from_quat(controller.mj_data.qpos[3:7])
    yaw_unwrap = 0.0          # accumulated signed yaw since step_start
    last_yaw = step_start_yaw
    min_distance = float("inf")  # closest approach this sub-step (m)

    print(f"[main] instruction = {args.instruction!r}")
    print(f"[main] policy_dt={policy_dt}s, render={args.render_hz} Hz, "
          f"max_sim={args.max_sim_seconds}s")
    if vlm is not None:
        print(f"[main] starting sub-step 1: {substeps[0].instruction!r}")

    try:
        while controller.is_running:
            t_loop = time.perf_counter()

            # --- (1a) read VLM command -----------------------------------
            if vlm is not None:
                vx, vy, vyaw = vlm.get_command()
            else:
                vx, vy, vyaw = const_cmd

            # --- (1b) sub-step termination check -------------------------
            if vlm is not None and drain_deadline is None:
                cur_yaw = yaw_from_quat(controller.mj_data.qpos[3:7])
                # Accumulate signed yaw delta (handles >180° turns).
                yaw_unwrap += wrap_pi(cur_yaw - last_yaw)
                last_yaw = cur_yaw

                cur = substeps[step_idx]
                done_reason = None

                # 1) yaw target (most reliable for turn commands)
                if cur.yaw_delta_target is not None:
                    tgt = cur.yaw_delta_target
                    yaw_tol = math.radians(args.turn_tolerance_deg)
                    if (tgt > 0 and yaw_unwrap >= tgt - yaw_tol) or \
                       (tgt < 0 and yaw_unwrap <= tgt + yaw_tol):
                        done_reason = (
                            f"yaw target reached "
                            f"(Δ={math.degrees(yaw_unwrap):+.0f}°, "
                            f"target={math.degrees(tgt):+.0f}°, "
                            f"tol=±{args.turn_tolerance_deg:.0f}°)"
                        )

                # 2) proximity target — direct hit OR closest-approach overshoot
                if done_reason is None and cur.proximity_target is not None:
                    tx, ty = cur.proximity_target[0], cur.proximity_target[1]
                    rx, ry = controller.mj_data.qpos[0], controller.mj_data.qpos[1]
                    d = math.hypot(tx - rx, ty - ry)
                    if d < cur.proximity_threshold:
                        done_reason = f"reached target (d={d:.2f}m)"
                    elif (min_distance < args.closest_approach_min and
                          d > min_distance + args.closest_approach_margin):
                        done_reason = (
                            f"closest approach passed "
                            f"(min={min_distance:.2f}m, now={d:.2f}m)"
                        )
                    min_distance = min(min_distance, d)

                # 3) NaVILA emitted "stop"
                if done_reason is None and vlm.stop_event.is_set():
                    done_reason = "NaVILA stop"

                # 4) per-step time limit
                step_elapsed = time.perf_counter() - step_start_time
                if done_reason is None and step_elapsed >= cur.time_limit:
                    done_reason = f"time limit ({cur.time_limit:.0f}s)"

                if done_reason is not None:
                    print(f"[plan] sub-step {step_idx + 1}/{len(substeps)} done: "
                          f"{done_reason}", flush=True)
                    step_idx += 1
                    if step_idx >= len(substeps):
                        print("[plan] all sub-steps complete — draining 2 s")
                        drain_deadline = time.perf_counter() + 2.0
                    else:
                        next_ss = substeps[step_idx]
                        print(f"[plan] starting sub-step {step_idx + 1}/{len(substeps)}: "
                              f"{next_ss.instruction!r}")
                        step_start_time = time.perf_counter()
                        step_start_yaw = cur_yaw
                        last_yaw = cur_yaw
                        yaw_unwrap = 0.0
                        min_distance = float("inf")
                        vlm.clear_stop()
                        vlm.set_instruction(next_ss.instruction)
                        # Use the just-cleared command so the K1 doesn't
                        # carry the previous step's velocity into this one
                        # while waiting on the next NaVILA result.
                        vx, vy, vyaw = 0.0, 0.0, 0.0

            # --- (1c) per-sub-step low-level controller -----------------
            # Two mutually-exclusive controllers, both based on the
            # sub-step type that parse_substeps detected.
            #
            # (i) Pure-turn sub-step ('turn left/right N deg'): NaVILA
            #     stubbornly emits "move forward" most of the time even
            #     when asked to turn, so the previous run's "turn" was
            #     actually a 4 m × 4 m drift arc that happened to hit
            #     -90° of yaw. We bypass NaVILA entirely and feed the
            #     walking policy (vx=0, vy=0, vyaw = K * (target -
            #     yaw_unwrap)). Existing yaw-target termination still
            #     fires when the turn is complete.
            #
            # (ii) Proximity sub-step: heading assist overlays vyaw =
            #      K * bearing_err on top of NaVILA's vx so the K1
            #      keeps pointing at the target while NaVILA controls
            #      forward speed.
            heading_used = False
            turn_used = False
            if (vlm is not None and drain_deadline is None
                    and step_idx < len(substeps)):
                cur_ss = substeps[step_idx]
                pure_turn = (cur_ss.yaw_delta_target is not None
                              and cur_ss.proximity_target is None)
                if pure_turn and args.turn_controller:
                    remaining = cur_ss.yaw_delta_target - yaw_unwrap
                    vx = 0.0
                    vy = 0.0
                    # P controller with a floor on |vyaw| — the walking
                    # policy needs at least ~0.25 rad/s to actually rotate
                    # the base; below that the gait stays in place and we
                    # asymptote forever shy of the target.
                    sign = 1.0 if remaining >= 0.0 else -1.0
                    mag = max(args.turn_min_vyaw,
                               abs(args.turn_kp * remaining))
                    vyaw = sign * min(args.vyaw_max, mag)
                    turn_used = True
                elif cur_ss.proximity_target is not None and args.heading_assist:
                    tx, ty = cur_ss.proximity_target[0], cur_ss.proximity_target[1]
                    rx, ry = controller.mj_data.qpos[0], controller.mj_data.qpos[1]
                    cur_yaw_now = yaw_from_quat(controller.mj_data.qpos[3:7])
                    target_bearing = math.atan2(ty - ry, tx - rx)
                    bearing_err = wrap_pi(target_bearing - cur_yaw_now)
                    assist = max(-args.vyaw_max,
                                  min(args.vyaw_max, args.heading_kp * bearing_err))
                    vyaw = max(-args.vyaw_max,
                                min(args.vyaw_max, vyaw + assist))
                    heading_used = True

            if drain_deadline is not None:
                vx, vy, vyaw = 0.0, 0.0, 0.0

            controller.vel_command.lin_vel_x = float(vx)
            controller.vel_command.lin_vel_y = float(vy)
            controller.vel_command.ang_vel_yaw = float(vyaw)

            # --- (2) one walking-policy step (also runs decimation × physics)
            controller.update_state()
            try:
                dof_targets = controller.policy_step()
            except Exception as e:
                print(f"[main] policy step failed: {e!r}")
                break
            controller.ctrl_step(dof_targets)
            if not controller.is_running:
                print("[main] safety fallback fired (robot fell). Exiting.")
                break

            # --- (3) head-cam frame for the VLM buffer -------------------
            now = time.perf_counter()
            if vlm is not None and now >= next_head_push:
                head_rgb = render_camera(head_renderer, controller.mj_data, "head_cam")
                vlm.push_frame(head_rgb)
                next_head_push = now + head_buffer_period

            # --- (4) video render at render_hz ---------------------------
            if now >= next_render:
                head_rgb = render_camera(head_renderer, controller.mj_data, "head_cam")
                scene_rgb = render_camera(scene_renderer, controller.mj_data, "scene_cam")

                # HUD
                pos = controller.mj_data.qpos[0:3].copy()
                quat = controller.mj_data.qpos[3:7]
                yaw_deg = float(np.degrees(yaw_from_quat(quat)))
                tgt = np.array(args.target[:2])
                dxy = tgt - pos[:2]
                dist = float(np.hypot(*dxy))
                bearing = float(np.degrees(math.atan2(dxy[1], dxy[0]) -
                                            math.atan2(2.0 * (quat[0] * quat[3] +
                                                                quat[1] * quat[2]),
                                                       1.0 - 2.0 * (quat[2] * quat[2] +
                                                                     quat[3] * quat[3]))))
                bearing = ((bearing + 180.0) % 360.0) - 180.0  # wrap to [-180,180]

                if vlm is not None:
                    s = vlm.status()
                    cur_ss = substeps[min(step_idx, len(substeps) - 1)]
                    plan_line = f"sub-step {step_idx + 1}/{len(substeps)}: {cur_ss.instruction[:60]}"
                    cond_line = ""
                    if cur_ss.yaw_delta_target is not None:
                        cond_line = (
                            f"yaw Δ={math.degrees(yaw_unwrap):+.0f}° / "
                            f"{math.degrees(cur_ss.yaw_delta_target):+.0f}°"
                        )
                    elif cur_ss.proximity_target is not None:
                        tx, ty = cur_ss.proximity_target[0], cur_ss.proximity_target[1]
                        d = math.hypot(tx - pos[0], ty - pos[1])
                        mind = (f" min={min_distance:.2f}m"
                                if min_distance != float("inf") else "")
                        cond_line = (f"distance {d:.2f}m / "
                                     f"{cur_ss.proximity_threshold:.1f}m{mind}")
                    else:
                        rem = cur_ss.time_limit - (now - step_start_time)
                        cond_line = f"time remaining {max(0, rem):.0f}s"
                    if turn_used:
                        assist_tag = " +turn-controller"
                    elif heading_used:
                        assist_tag = " +heading"
                    else:
                        assist_tag = ""
                    hud_lines = [
                        plan_line,
                        f"  {cond_line}",
                        f"vlm #{s['inf_count']:03d} {s['inf_ms']:.0f}ms  {s['label']}",
                        f"cmd  vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f}{assist_tag}",
                        f"pose ({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:.2f}) "
                        f"yaw={yaw_deg:+.0f}°  red dist={dist:.2f}m",
                    ]
                else:
                    hud_lines = [
                        f"--no-vlm  cmd vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f}",
                        f"pose ({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:.2f}) "
                        f"yaw={yaw_deg:+.0f}°",
                        f"red dist={dist:.2f}m bearing={bearing:+.0f}°",
                    ]
                scene_bgr = draw_hud(cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR),
                                       hud_lines)
                head_bgr = cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)

                if scene_writer is not None:
                    scene_writer.write(scene_bgr)
                    head_writer.write(head_bgr)
                next_render = now + render_period

            # --- (5) periodic console line -------------------------------
            if now >= next_print:
                if vlm is not None:
                    s = vlm.status()
                    yaw_now = math.degrees(yaw_from_quat(controller.mj_data.qpos[3:7]))
                    tag = ("TURN" if turn_used
                           else "HEAD" if heading_used else "VLM ")
                    print(f"[main t={now - sim_t0:5.1f}s] "
                          f"applied[{tag}] vx={vx:+.2f} vy={vy:+.2f} "
                          f"vyaw={vyaw:+.2f}  "
                          f"vlm({s['vx']:+.2f},{s['vy']:+.2f},{s['vyaw']:+.2f}) "
                          f"pose=({controller.mj_data.qpos[0]:+.2f},"
                          f"{controller.mj_data.qpos[1]:+.2f},"
                          f"{controller.mj_data.qpos[2]:.2f}) "
                          f"yaw={yaw_now:+.0f}° yawΔ={math.degrees(yaw_unwrap):+.0f}°",
                          flush=True)
                else:
                    print(f"[main t={now - sim_t0:5.1f}s] "
                          f"pose=({controller.mj_data.qpos[0]:+.2f},"
                          f"{controller.mj_data.qpos[1]:+.2f},"
                          f"{controller.mj_data.qpos[2]:.2f})", flush=True)
                next_print = now + 1.0

            # --- (6) terminate? ------------------------------------------
            if drain_deadline is not None and time.perf_counter() >= drain_deadline:
                print("[main] drain complete — exiting.")
                break
            if vlm is not None and vlm.status()["inf_count"] >= args.steps:
                # Reached step budget; allow drain.
                if drain_deadline is None:
                    print(f"[main] reached --steps={args.steps}, draining.")
                    drain_deadline = time.perf_counter() + 2.0
            if (now - sim_t0) > args.max_sim_seconds:
                print("[main] hit --max-sim-seconds, exiting.")
                break

            # --- (7) pace the loop to policy_dt --------------------------
            elapsed = time.perf_counter() - t_loop
            sleep = policy_dt - elapsed
            if sleep > 0:
                time.sleep(sleep)
    finally:
        print("[main] cleanup ...")
        if vlm is not None:
            vlm.shutdown()
        if scene_writer is not None:
            scene_writer.release()
        if head_writer is not None:
            head_writer.release()
        head_renderer.close()
        scene_renderer.close()
        # final pose summary
        pos = controller.mj_data.qpos[0:3]
        print(f"[main] final pose = ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:.2f})")


# ============================================================================
# helpers
# ============================================================================

def render_camera(renderer: mujoco.Renderer, data: mujoco.MjData,
                  cam_name: str) -> np.ndarray:
    renderer.update_scene(data, camera=cam_name)
    return renderer.render()


def draw_hud(bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    y = 24
    for line in lines:
        cv2.putText(bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return bgr


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y),
                              1.0 - 2.0 * (y * y + z * z)))


if __name__ == "__main__":
    main()
