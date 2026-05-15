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
    NUM_FRAMES, ACTION_DURATION, build_prompt, parse_action,
)


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
) -> Optional[str]:
    """Return a human-readable reason if this sub-step is done, else None.

    ``current_pos_xy`` may be ``None`` when no pose source exists (real
    robot without odometry); proximity / closest-approach conditions are
    then skipped.
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
        # Episode-length frame log. Paper §II-A: the first frame is ALWAYS
        # included as part of the historical context, and the other 6
        # historical frames are uniformly sampled across the episode — so
        # we cannot use a fixed-size FIFO that evicts the start. Reset on
        # session start (see Session.__init__ / navila_server.py).
        self._frame_buffer: list = []
        # Defensive memory cap. At higher frame rates / long episodes the
        # buffer would otherwise grow unbounded (5 min @ 3fps × 720p RGB
        # ≈ 2-3 GB). Once we exceed ``_buf_soft_cap`` frames, compact by
        # keeping frame[0], the most recent ``_buf_recent_keep`` frames,
        # and a uniformly-sampled slice of the middle — this preserves
        # the linspace-sampling invariant the inference path relies on.
        self._buf_soft_cap = 500
        self._buf_recent_keep = 50

        self.stop_event = threading.Event()
        self._abort = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --- model load (heavy; deferred so tests can skip) -----------------

    def load_model(self) -> None:
        from llava.model.builder import load_pretrained_model  # noqa: E402
        from llava.mm_utils import get_model_name_from_path  # noqa: E402
        from llava.utils import disable_torch_init  # noqa: E402
        disable_torch_init()
        print(f"[VLM] loading NaVILA from {self.model_path} ...", flush=True)
        model_name = get_model_name_from_path(str(self.model_path))
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            str(self.model_path), model_name, model_base=None,
            attn_implementation="sdpa",
        )
        print("[VLM] NaVILA ready.", flush=True)

    def set_inference_fn(self, fn) -> None:
        """Inject a fake inference function (for tests)."""
        self._inference_fn = fn

    # --- API used by main thread ----------------------------------------

    def push_frame(self, rgb) -> None:
        from PIL import Image  # local import to keep tests light
        if not isinstance(rgb, Image.Image):
            rgb = Image.fromarray(rgb)
        with self._frame_lock:
            self._frame_buffer.append(rgb)
            if len(self._frame_buffer) > self._buf_soft_cap:
                self._frame_buffer = self._compact_buffer(
                    self._frame_buffer,
                    cap=self._buf_soft_cap,
                    recent_keep=self._buf_recent_keep,
                )

    @staticmethod
    def _compact_buffer(buffer: list, cap: int, recent_keep: int) -> list:
        """Reduce a too-large frame buffer in-place-safe.

        Keeps ``buffer[0]`` (paper §II-A invariant), the last
        ``recent_keep`` frames, and a uniformly-sampled slice of the
        middle to total ``cap`` frames.
        """
        if len(buffer) <= cap:
            return buffer
        first = buffer[0]
        recent = buffer[-recent_keep:]
        middle = buffer[1:-recent_keep]
        middle_budget = max(0, cap - 1 - recent_keep)
        if middle and middle_budget < len(middle):
            idx = np.linspace(0, len(middle) - 1,
                               num=middle_budget, dtype=int)
            middle = [middle[i] for i in idx]
        return [first] + middle + recent

    def bootstrap_buffer(self, rgb) -> None:
        """Seed the episode with the first frame.

        Paper §II-A pins the first frame in the historical context, so we
        only push it ONCE — uniform sampling at inference time fills the
        rest. (Previous implementations pre-filled NUM_FRAMES copies; that
        bloated the buffer and shifted the linspace sampling toward the
        bootstrap frame for the rest of the episode.)
        """
        from PIL import Image
        img = rgb if isinstance(rgb, Image.Image) else Image.fromarray(rgb)
        with self._frame_lock:
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

    @staticmethod
    def sample_frames(frames: list, num_frames: int = NUM_FRAMES) -> list:
        """Paper-correct frame selection for one NaVILA inference call.

        Per NaVILA paper §II-A, the prompt carries 8 frames for real-world
        deployment (Table IX): the latest frame is the "current
        observation"; the other 7 are historical frames uniformly sampled
        from the preceding t-1 frames, with the first frame ALWAYS
        included. Mirrors ``llava/mm_utils.get_frame_from_vcap_vlnce``:

            sampled = np.linspace(0, len(frames)-1, num=N-1,
                                   endpoint=False, dtype=int)
            out = [frames[i] for i in sampled] + [frames[-1]]

        ``np.linspace(... endpoint=False)`` always starts at 0, so
        ``frames[0]`` is in every sample. If the episode has fewer than
        ``num_frames`` frames so far, left-pad with the first frame (the
        existing convention; the reference repo pads with black frames
        which is harsher for the bootstrap window).
        """
        if not frames:
            raise ValueError("sample_frames called with empty buffer")
        if len(frames) < num_frames:
            pad = [frames[0]] * (num_frames - len(frames))
            return pad + list(frames)
        indices = np.linspace(0, len(frames) - 1,
                               num=num_frames - 1,
                               endpoint=False, dtype=int)
        return [frames[i] for i in indices] + [frames[-1]]

    def _sample_frames_for_inference(self) -> list:
        return self.sample_frames(self._snapshot(), NUM_FRAMES)

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
            frames = self._sample_frames_for_inference()
            instruction = self.get_instruction()
            t0 = time.perf_counter()
            try:
                raw = self._do_inference(frames, instruction)
            except Exception as e:
                print(f"[VLM] inference failed: {e!r}", flush=True)
                time.sleep(0.5)
                continue
            inf_ms = (time.perf_counter() - t0) * 1000.0

            vx, vy, vyaw, duration, label = parse_action(raw, self.action_duration)
            req_vx, req_vy, req_vyaw = vx, vy, vyaw
            vx = max(-self.vx_max, min(self.vx_max, vx))
            vy = max(-self.vy_max, min(self.vy_max, vy))
            vyaw = max(-self.vyaw_max, min(self.vyaw_max, vyaw))

            # If the operator's safety cap clipped the paper-spec command
            # speed, stretch the hold so the requested distance / angle is
            # still achieved (motion = clipped_speed × stretched_duration
            # ≡ paper_speed × paper_duration). Keeps the model's intent
            # under conservative caps; only deviation from paper is the
            # slightly longer execution time.
            if duration > 0.0:
                scale = 1.0
                for req, applied in ((req_vx, vx), (req_vy, vy),
                                      (req_vyaw, vyaw)):
                    if applied != 0.0 and abs(applied) < abs(req):
                        scale = max(scale, abs(req) / abs(applied))
                duration *= scale

            self._publish(vx, vy, vyaw, label, raw, inf_ms)
            print(f"[VLM #{self._inf_count:03d} {inf_ms:5.0f}ms] "
                  f"task={instruction!r}\n"
                  f"            raw={raw!r}\n"
                  f"            -> {label}  "
                  f"vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f} "
                  f"hold={duration:.2f}s", flush=True)

            if label == "stop":
                self.stop_event.set()
                # Avoid busy-spinning inference while the planner advances
                # the sub-step / enters drain.
                time.sleep(0.1)
                continue

            # Paper §II-B: hold the fixed velocity for the action's
            # duration, then zero so the robot settles before we re-ask.
            if duration > 0.0:
                deadline = t0 + (inf_ms / 1000.0) + duration
                while not self._abort.is_set():
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    time.sleep(min(0.05, remaining))
                if self._abort.is_set():
                    break
                self._publish(0.0, 0.0, 0.0, f"{label} (settle)", raw, 0.0)
                # Brief settle so the K1 walker comes to rest before the
                # next VLM frame capture / inference cycle.
                time.sleep(0.1)


def _navila_inference(tokenizer, model, image_processor,
                      frames: list, instruction: str,
                      max_new_tokens: int = 256) -> str:
    """The actual NaVILA call. Module-level so VLMRunner can be tested
    without importing torch / llava.

    ``max_new_tokens=256`` matches the reference inference script
    (``booster/NaVILA/llava/eval/run_navigation.py`` uses 1024 as the
    arg default and 1024 hard-coded in the call). NaVILA emits a short
    scene description followed by the action, so a 64-token cap can
    silently truncate before the action verb appears and force the
    parser into "unparsed -> stop" — i.e. the robot freezes."""
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
