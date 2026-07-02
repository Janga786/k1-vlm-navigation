"""Reusable core: multi-step planner + NaVILA runner + helpers.

Imported by both:
- navila_k1_walking_loop.py  — MuJoCo sim2sim with the trained velocity policy
- navila_k1_realrobot.py     — K1 SDK with the built-in walker (B1LocoClient)

Contains nothing MuJoCo-specific or SDK-specific so it's testable in isolation
(see tests/ — the parser / planner-logic tests need no NaVILA, no MuJoCo, no
booster_robotics_sdk_python).
"""

from __future__ import annotations

import math
import os
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# --- path injection so we can import llava regardless of editable-install state
_K1RES = Path.home() / "Projects" / "k1_research"
_NAVILA_REPO = _K1RES / "booster" / "NaVILA"
for p in (_NAVILA_REPO,):
    sp = str(p)
    if sp not in sys.path and p.exists():
        sys.path.insert(0, sp)


# Re-exported from the bridge so callers don't need a second import.
sys.path.insert(0, str(Path(__file__).parent))
from navila_k1_bridge import (  # noqa: E402
    NUM_FRAMES, ACTION_DURATION, build_prompt, parse_action, parse_action_hold,
)


# ============================================================================
# Camera input alignment: make the on-robot frame match the sim benchmark's
# distortion-free pinhole + chosen squarify, so NaVILA sees the same image
# distribution on the robot and in eval. Override at runtime (no code edit):
#   NAVILA_VLM_TRANSFORM = stretch | crop | pad   (MUST match the benchmark run)
#   NAVILA_UNDISTORT     = 1 (default) | 0
# BoosterMipi head-cam factory intrinsics @1280x720 (read on-robot 2026-06-08
# from the soccer vision.yaml SN-calibration). The sim camera is a perfect
# pinhole, so we undistort the real barrel (k1=-0.055..) to match it.
# ============================================================================
VLM_TRANSFORM = os.environ.get("NAVILA_VLM_TRANSFORM", "stretch")
UNDISTORT = os.environ.get("NAVILA_UNDISTORT", "1") != "0"
_MIPI_K = np.array([[643.898, 0.0, 649.038],
                    [0.0, 643.216, 357.21],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
_MIPI_DIST = np.array([-0.0553056, 0.065975, -0.000994232, 2.98548e-05, -0.0216579],
                      dtype=np.float64)
_MIPI_CAL_W, _MIPI_CAL_H = 1280, 720


def _squarify(img, mode):
    """PIL RGB -> square PIL. stretch=no-op (the VLM resize distorts 16:9->1:1),
    crop=center-crop (undistorted, ~58deg), pad=letterbox (undistorted, full
    ~90deg HFOV, black bars)."""
    from PIL import Image
    if mode == "stretch":
        return img
    W, H = img.size
    if W == H:
        return img
    if mode == "crop":
        s = min(W, H)
        l, t = (W - s) // 2, (H - s) // 2
        return img.crop((l, t, l + s, t + s))
    s = max(W, H)
    canvas = Image.new("RGB", (s, s), (0, 0, 0))
    canvas.paste(img, ((s - W) // 2, (s - H) // 2))
    return canvas


def prep_frame(rgb):
    """Robot-side camera alignment: undistort the BoosterMipi barrel + squarify so
    the frame NaVILA sees on the real robot matches the sim benchmark. Use the
    SAME transform mode as the benchmark run (env NAVILA_VLM_TRANSFORM)."""
    from PIL import Image
    arr = np.asarray(rgb.convert("RGB")) if isinstance(rgb, Image.Image) else np.asarray(rgb)
    if UNDISTORT:
        try:
            import cv2
            h, w = arr.shape[:2]
            K = _MIPI_K.copy()
            if (w, h) != (_MIPI_CAL_W, _MIPI_CAL_H):  # scale intrinsics to actual frame size
                K[0] *= w / _MIPI_CAL_W
                K[1] *= h / _MIPI_CAL_H
            arr = cv2.undistort(arr, K, _MIPI_DIST)
        except Exception as e:  # cv2 missing / odd frame -> keep raw, don't crash deploy
            print(f"[prep_frame] undistort skipped: {e!r}", flush=True)
    return _squarify(Image.fromarray(arr), VLM_TRANSFORM)


# ============================================================================
# Multi-step planner
# ============================================================================


@dataclass
class SubStep:
    """One atomic instruction handed to NaVILA, plus how we know it is done.

    Termination priority (first match wins):
      1. ``yaw_delta_target`` reached (signed radians, accumulated since the
         start of this sub-step).  ±tolerance applied (see main loops).
      2. ``proximity_target`` reached within ``proximity_threshold`` (m, xy).
      3. NaVILA emits "stop".
      4. ``time_limit`` seconds elapsed.

    Pose-based conditions (yaw, proximity) are skipped when the entry point
    has no pose source (e.g. real robot without odometry); only stop + time
    apply in that case.
    """
    instruction: str
    time_limit: float = 30.0
    proximity_target: Optional[tuple[float, float, float]] = None
    proximity_threshold: float = 1.0
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
    outside, which is more reliable than waiting for NaVILA's "stop".

    - ``"turn (left|right) N deg"``  → yaw-delta termination
    - mentions of a known scene target (e.g. "red box") → proximity
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


# ============================================================================
# Per-sub-step termination
# ============================================================================


@dataclass
class TerminationState:
    """Mutable state tracked across one sub-step."""
    step_idx: int
    started_at: float
    start_yaw: float
    last_yaw: float
    yaw_unwrap: float = 0.0          # signed accumulated yaw (rad)
    min_distance: float = float("inf")  # closest approach (m)
    # Windowed-progress stall backstop (needs pose; see check_termination).
    stall_anchor_xy: Optional[tuple[float, float]] = None
    stall_anchor_yaw: float = 0.0    # yaw_unwrap at the anchor
    stall_anchor_t: float = 0.0

    def reset_stall_anchor(self) -> None:
        self.stall_anchor_xy = None
        self.stall_anchor_yaw = 0.0
        self.stall_anchor_t = 0.0


def update_yaw_unwrap(state: TerminationState, current_yaw: float) -> None:
    """Add the wrapped delta to the running unwrapped yaw and update last_yaw."""
    state.yaw_unwrap += wrap_pi(current_yaw - state.last_yaw)
    state.last_yaw = current_yaw


def check_termination(
    ss: SubStep,
    state: TerminationState,
    *,
    current_pos_xy: Optional[tuple[float, float]],
    vlm_stop: bool,
    now: float,
    yaw_tolerance_deg: float = 5.0,
    closest_approach_min: float = 1.5,
    closest_approach_margin: float = 0.25,
    cmd_moving: bool = False,
    stall_window_s: float = 10.0,
    stall_disp_m: float = 0.25,
    stall_yaw_deg: float = 30.0,
) -> Optional[str]:
    """Return a human-readable reason if this sub-step is done, else None.

    ``current_pos_xy`` may be ``None`` when no pose source exists (real
    robot without odometry); proximity / closest-approach / stall
    conditions are then skipped.

    ``cmd_moving`` (caller passes whether a non-zero velocity is actually
    being sent) arms the stall backstop: if the robot is commanded to move
    but covers < ``stall_disp_m`` AND rotates < ``stall_yaw_deg`` over a
    ``stall_window_s`` window, the sub-step is aborted instead of grinding
    against the obstacle until the time limit (thresholds = the validated
    benchmark hopeless-stall gate: wedge 0.13m/8deg fires, escape-grind
    0.50m/37deg does not).
    """
    # 1) yaw target — most reliable for turn commands
    if ss.yaw_delta_target is not None:
        tgt = ss.yaw_delta_target
        tol = math.radians(yaw_tolerance_deg)
        if (tgt > 0 and state.yaw_unwrap >= tgt - tol) or \
           (tgt < 0 and state.yaw_unwrap <= tgt + tol):
            return (f"yaw target reached "
                    f"(Δ={math.degrees(state.yaw_unwrap):+.0f}°, "
                    f"target={math.degrees(tgt):+.0f}°, "
                    f"tol=±{yaw_tolerance_deg:.0f}°)")

    # 2) proximity / closest-approach (require pose)
    if ss.proximity_target is not None and current_pos_xy is not None:
        tx, ty = ss.proximity_target[0], ss.proximity_target[1]
        rx, ry = current_pos_xy
        d = math.hypot(tx - rx, ty - ry)
        if d < ss.proximity_threshold:
            return f"reached target (d={d:.2f}m)"
        if (state.min_distance < closest_approach_min and
                d > state.min_distance + closest_approach_margin):
            reason = (f"closest approach passed "
                      f"(min={state.min_distance:.2f}m, now={d:.2f}m)")
            state.min_distance = min(state.min_distance, d)
            return reason
        state.min_distance = min(state.min_distance, d)

    # 3) NaVILA stop
    if vlm_stop:
        return "NaVILA stop"

    # 3.5) windowed-progress stall backstop (needs pose + an active command).
    # A wedged biped wobbles (~1 rad/s base noise) and creeps, so instantaneous
    # velocity gates false-fire; only NET progress over a window discriminates.
    if current_pos_xy is not None and cmd_moving:
        if state.stall_anchor_xy is None:
            state.stall_anchor_xy = current_pos_xy
            state.stall_anchor_yaw = state.yaw_unwrap
            state.stall_anchor_t = now
        else:
            disp = math.hypot(current_pos_xy[0] - state.stall_anchor_xy[0],
                              current_pos_xy[1] - state.stall_anchor_xy[1])
            dyaw = abs(state.yaw_unwrap - state.stall_anchor_yaw)
            if disp >= stall_disp_m or dyaw >= math.radians(stall_yaw_deg):
                # Progressing — slide the window forward.
                state.stall_anchor_xy = current_pos_xy
                state.stall_anchor_yaw = state.yaw_unwrap
                state.stall_anchor_t = now
            elif (now - state.stall_anchor_t) >= stall_window_s:
                return (f"stall-abort ({disp:.2f}m / "
                        f"{math.degrees(dyaw):.0f}deg net over "
                        f"{stall_window_s:.0f}s of commanded motion)")
    else:
        # Not commanding motion (or no pose) — a standing robot is not stalled.
        state.stall_anchor_xy = None

    # 4) time limit
    if (now - state.started_at) >= ss.time_limit:
        return f"time limit ({ss.time_limit:.0f}s)"

    return None


# ============================================================================
# Inner-loop controllers (heading-assist + open-loop turn)
# ============================================================================


@dataclass
class ControllerOutput:
    """The (vx, vy, vyaw) we'll actually send and a tag for HUD/logs."""
    vx: float
    vy: float
    vyaw: float
    tag: str  # "VLM" / "HEAD" / "TURN"


def apply_controllers(
    *,
    ss: SubStep,
    state: TerminationState,
    current_pos_xy: Optional[tuple[float, float]],
    current_yaw: Optional[float],
    vlm_cmd: tuple[float, float, float],
    vx_max: float,
    vy_max: float,
    vyaw_max: float,
    heading_assist: bool = True,
    heading_kp: float = 1.5,
    turn_controller: bool = True,
    turn_kp: float = 2.0,
    turn_min_vyaw: float = 0.30,
) -> ControllerOutput:
    """Decide what to actually send to the actuator this tick.

    Three cases (mutually exclusive):
    - **TURN**: pure-turn sub-step + turn_controller on → bypass VLM
      vx/vy/vyaw, drive ``vx=vy=0, vyaw = clip(K·(target − unwrap))`` with
      a min-magnitude floor.
    - **HEAD**: proximity sub-step + heading_assist on + pose available →
      keep VLM's vx/vy and overlay
      ``vyaw += K·(bearing − yaw)`` clipped to ``vyaw_max``.
    - **VLM**: passthrough (clipped).
    """
    vx, vy, vyaw = vlm_cmd
    pure_turn = (ss.yaw_delta_target is not None
                  and ss.proximity_target is None)

    if pure_turn and turn_controller:
        remaining = ss.yaw_delta_target - state.yaw_unwrap
        sign = 1.0 if remaining >= 0.0 else -1.0
        mag = max(turn_min_vyaw, abs(turn_kp * remaining))
        vyaw = sign * min(vyaw_max, mag)
        return ControllerOutput(vx=0.0, vy=0.0, vyaw=vyaw, tag="TURN")

    if (ss.proximity_target is not None and heading_assist
            and current_pos_xy is not None and current_yaw is not None):
        tx, ty = ss.proximity_target[0], ss.proximity_target[1]
        rx, ry = current_pos_xy
        target_bearing = math.atan2(ty - ry, tx - rx)
        bearing_err = wrap_pi(target_bearing - current_yaw)
        assist = max(-vyaw_max, min(vyaw_max, heading_kp * bearing_err))
        vyaw = max(-vyaw_max, min(vyaw_max, vyaw + assist))
        vx = max(-vx_max, min(vx_max, vx))
        vy = max(-vy_max, min(vy_max, vy))
        return ControllerOutput(vx=vx, vy=vy, vyaw=vyaw, tag="HEAD")

    vx = max(-vx_max, min(vx_max, vx))
    vy = max(-vy_max, min(vy_max, vy))
    vyaw = max(-vyaw_max, min(vyaw_max, vyaw))
    return ControllerOutput(vx=vx, vy=vy, vyaw=vyaw, tag="VLM ")


# ============================================================================
# VLM runner (background thread)
# ============================================================================


class VLMRunner:
    """NaVILA inference loop on a rolling 8-frame head-camera buffer.

    Main thread pushes frames and reads (vx, vy, vyaw); this thread spends
    most of its time inside model.generate(). When NaVILA emits "stop"
    ``stop_event`` is set; main loop is responsible for clearing it on
    sub-step advance.

    Construction is split from start() so tests can substitute a fake
    inference function.
    """

    def __init__(
        self,
        model_path: Path,
        action_duration: float = ACTION_DURATION,
        vx_max: float = 0.6,
        vy_max: float = 0.3,
        vyaw_max: float = 0.6,
    ):
        self.action_duration = action_duration
        self.vx_max, self.vy_max, self.vyaw_max = vx_max, vy_max, vyaw_max
        self.model_path = model_path

        self.tokenizer = None
        self.model = None
        self.image_processor = None
        # Caller can set this to a custom inference function for testing
        # (signature: frames: list[PIL.Image] -> str).
        self._inference_fn = None

        self._lock = threading.Lock()
        self._cmd = (0.0, 0.0, 0.0)
        self._label = "(waiting on first vlm result)"
        self._raw_text = ""
        self._inf_count = 0
        self._inf_ms = 0.0

        self._instruction_lock = threading.Lock()
        self._instruction = "(none)"

        self._frame_lock = threading.Lock()
        self._frame_buffer: deque = deque(maxlen=NUM_FRAMES)

        self.stop_event = threading.Event()
        self._abort = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --- model load (heavy; deferred so tests can skip) -----------------

    def load_model(self, load_8bit: bool = False) -> None:
        # The VILA-aware llava (experiments/navila) MUST win the import over any
        # standard LLaVA in ~/.local user-site (which shadows conda env packages and
        # cannot read VILA checkpoints — tokenizer/LLM live under llm/). Same
        # bootstrap as vlm_server_bridge.py (2026-06-05 lab fix).
        import sys
        from pathlib import Path as _P
        _vila = os.environ.get(
            "NAVILA_VILA_REPO",
            str(_P.home() / "Projects/k1_research/experiments/navila"))
        if _P(_vila).exists() and _vila not in sys.path:
            sys.path.insert(0, _vila)
        from llava.model.builder import load_pretrained_model  # noqa: E402
        from llava.mm_utils import get_model_name_from_path  # noqa: E402
        from llava.utils import disable_torch_init  # noqa: E402
        disable_torch_init()
        print(f"[VLM] loading NaVILA from {self.model_path} "
              f"({'int8' if load_8bit else 'fp16'}) ...", flush=True)
        model_name = get_model_name_from_path(str(self.model_path))
        # installed VILA llava signature is (model_path, model_base, model_name, ...);
        # passing model_name positionally lands it in model_base -> TypeError
        # (same fix as vlm_server_bridge.py, 2026-06-05)
        kwargs = {"attn_implementation": "sdpa"}
        if load_8bit:
            # int8 = ~9GB vs ~17GB fp16; benchmark-blessed (replay arbiter:
            # 7.1% action drift, 100% agreement on success episodes).
            kwargs["load_8bit"] = True
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            str(self.model_path), model_base=None, model_name=model_name,
            **kwargs,
        )
        print("[VLM] NaVILA ready.", flush=True)

    def set_inference_fn(self, fn) -> None:
        """Inject a fake inference function (for tests)."""
        self._inference_fn = fn

    # --- API used by main thread ----------------------------------------

    def push_frame(self, rgb) -> None:
        rgb = prep_frame(rgb)  # undistort + squarify to match the sim benchmark
        with self._frame_lock:
            self._frame_buffer.append(rgb)

    def bootstrap_buffer(self, rgb) -> None:
        img = prep_frame(rgb)  # undistort + squarify to match the sim benchmark
        with self._frame_lock:
            for _ in range(NUM_FRAMES):
                self._frame_buffer.append(img)

    def get_command(self) -> tuple[float, float, float]:
        with self._lock:
            return self._cmd

    def set_instruction(self, text: str) -> None:
        with self._instruction_lock:
            self._instruction = text
        with self._lock:
            self._cmd = (0.0, 0.0, 0.0)
            self._label = f"(switching to: {text})"

    def get_instruction(self) -> str:
        with self._instruction_lock:
            return self._instruction

    def clear_stop(self) -> None:
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
        if self._inference_fn is None and self.model is None:
            raise RuntimeError(
                "VLMRunner.start() called before load_model() and no "
                "inference function injected via set_inference_fn().")
        self._thread = threading.Thread(target=self._loop, name="vlm",
                                         daemon=True)
        self._thread.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        self._abort.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # --- internals ------------------------------------------------------

    def _snapshot(self) -> list:
        with self._frame_lock:
            return list(self._frame_buffer)

    def _publish(self, vx, vy, vyaw, label, raw, inf_ms):
        with self._lock:
            self._cmd = (vx, vy, vyaw)
            self._label = label
            self._raw_text = raw
            self._inf_count += 1
            self._inf_ms = inf_ms

    def _do_inference(self, frames, instruction: str) -> str:
        """Override point: production uses NaVILA, tests inject a stub."""
        if self._inference_fn is not None:
            return self._inference_fn(frames, instruction)
        return _navila_inference(self.tokenizer, self.model,
                                  self.image_processor, frames, instruction)

    def _loop(self) -> None:
        # Wait until the main thread bootstraps the buffer AND sets an
        # instruction. Don't ask NaVILA to plan for "(none)".
        while not self._abort.is_set() and (
            len(self._snapshot()) == 0
            or self.get_instruction() == "(none)"
        ):
            time.sleep(0.05)

        while not self._abort.is_set():
            frames = self._snapshot()
            if len(frames) == 0:
                # Camera source died mid-session (e.g. stream decay): don't crash the
                # inference thread on frames[0] — idle until frames return. (2026-06-11:
                # this exact IndexError killed navila_server during a camera outage.)
                print("[VLM] frame buffer empty — waiting for camera...", flush=True)
                time.sleep(1.0)
                continue
            if len(frames) < NUM_FRAMES:
                frames = [frames[0]] * (NUM_FRAMES - len(frames)) + frames
            instruction = self.get_instruction()
            t0 = time.perf_counter()
            try:
                raw = self._do_inference(frames, instruction)
            except Exception as e:
                print(f"[VLM] inference failed: {e!r}", flush=True)
                time.sleep(0.5)
                continue
            inf_ms = (time.perf_counter() - t0) * 1000.0

            vx, vy, vyaw, label, hold_s = parse_action_hold(
                raw, self.action_duration)
            vx = max(-self.vx_max, min(self.vx_max, vx))
            vy = max(-self.vy_max, min(self.vy_max, vy))
            vyaw = max(-self.vyaw_max, min(self.vyaw_max, vyaw))

            self._publish(vx, vy, vyaw, label, raw, inf_ms)
            print(f"[VLM #{self._inf_count:03d} {inf_ms:5.0f}ms] "
                  f"task={instruction!r}\n"
                  f"            raw={raw!r}\n"
                  f"            -> {label}  "
                  f"vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f} "
                  f"hold={hold_s:.1f}s", flush=True)

            if label == "stop":
                self.stop_event.set()
                continue

            # Execute-to-completion (benchmark semantics): hold this command
            # for its computed duration before the next inference so a clipped
            # turn/forward finishes its commanded displacement. Bail out early
            # if the sub-step advances (instruction change) or we're shutting
            # down. If the runner caps re-clipped below the parse-level caps,
            # stretch the hold so displacement (speed x time) is preserved.
            p_vx, _, p_vyaw, _, _ = parse_action_hold(raw, self.action_duration)
            if abs(vyaw) > 1e-6 and abs(p_vyaw) > abs(vyaw):
                hold_s = min(hold_s * abs(p_vyaw) / abs(vyaw), 6.0)
            elif abs(vx) > 1e-6 and abs(p_vx) > abs(vx):
                hold_s = min(hold_s * abs(p_vx) / abs(vx), 6.0)
            hold_until = time.perf_counter() + hold_s
            while (not self._abort.is_set()
                   and time.perf_counter() < hold_until
                   and self.get_instruction() == instruction
                   and not self.stop_event.is_set()):
                time.sleep(0.05)


def _navila_inference(tokenizer, model, image_processor,
                      frames: list, instruction: str,
                      max_new_tokens: int = 64) -> str:
    """The actual NaVILA call. Module-level so VLMRunner can be tested
    without importing torch / llava."""
    import torch
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria, process_images, tokenizer_image_token,
    )
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
    with torch.inference_mode():
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
# Helpers
# ============================================================================


def wrap_pi(a: float) -> float:
    return float(((a + math.pi) % (2.0 * math.pi)) - math.pi)


def yaw_from_quat(q) -> float:
    """Yaw from a (w, x, y, z) quaternion."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return float(math.atan2(2.0 * (w * z + x * y),
                              1.0 - 2.0 * (y * y + z * z)))


# Default scene-target name → world-pos mapping used by both entry scripts.
# (Real-robot version overrides this with whatever targets the operator
# declares via CLI.)
DEFAULT_SCENE_TARGETS: dict[str, tuple[float, float, float]] = {
    "red box":   (3.0, 0.0, 0.30),
    "red cube":  (3.0, 0.0, 0.30),
    "red":       (3.0, 0.0, 0.30),
    "blue box":  (2.0, -1.5, 0.25),
    "blue cube": (2.0, -1.5, 0.25),
    "blue":      (2.0, -1.5, 0.25),
    "green box":  (1.5, 1.8, 0.25),
    "green cube": (1.5, 1.8, 0.25),
    "green":      (1.5, 1.8, 0.25),
}
