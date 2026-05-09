#!/usr/bin/env python3
"""Closed-loop NaVILA → real K1 (built-in walker via B1LocoClient).

This is the real-robot deployment path. NaVILA picks navigation actions
from the K1's head camera (ZED), the multi-step planner + controllers
turn them into (vx, vy, vyaw), and the K1's **built-in walker** —
``B1LocoClient.Move`` — executes them. The trained sim policy is NOT
used here: the built-in walker is battle-tested by Booster across 700+
robots, so the only real-world unknown is whether NaVILA produces
sensible commands from real ZED images.

Three operating modes
=====================
- ``--mode print``  no SDK, no robot. Just print parsed actions.
                    Useful for static-image / recorded-replay smoke tests.
- ``--mode dry``    SDK initialised + connected, NaVILA running, planner
                    active — but every Move() call is logged instead of
                    sent. Use this on the real robot before live to
                    validate NaVILA's outputs against the ZED's view.
- ``--mode live``   Send Move() at SEND_HZ. Switches to kWalking on
                    start, kDamping on exit. KILL the process to stop.

Image source (``--image-source``)
==================================
- ``zed``      live frames from the ZED SDK on the robot.
- ``mjpeg``    HTTP MJPEG stream (e.g. "http://k1.local:8080/stream").
- ``dir``      offline replay: read ``--image-dir/*.jpg`` in name order.
- ``static``   single image at ``--image-path``, repeated.

For pre-flight validation **without the K1 powered on**:

    # offline NaVILA-on-recorded-frames sanity:
    python navila_k1_realrobot.py --mode print \\
        --image-source dir --image-dir ./recorded_zed_frames \\
        --instruction "navigate to the red chair"

    # all-up dry-run on the actual robot, no motion:
    python navila_k1_realrobot.py --mode dry \\
        --image-source zed --net 192.168.0.10 \\
        --instruction "walk forward 3 meters | turn left 90 deg | walk forward"

For live deploy (requires the floor cleared):

    python navila_k1_realrobot.py --mode live \\
        --image-source zed --net 192.168.0.10 \\
        --instruction "walk to the chair | turn right 90 deg | walk forward"
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from PIL import Image

# Local imports — keep these last so any path surgery from .pth files runs first
sys.path.insert(0, str(Path(__file__).parent))
from navila_k1_core import (  # noqa: E402
    DEFAULT_SCENE_TARGETS,
    NUM_FRAMES,
    SubStep,
    TerminationState,
    VLMRunner,
    apply_controllers,
    check_termination,
    describe_substep,
    parse_substeps,
    update_yaw_unwrap,
    wrap_pi,
    yaw_from_quat,
)
from navila_k1_bridge import ACTION_DURATION  # noqa: E402

DEFAULT_CKPT = Path.home() / "Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f"


# ============================================================================
# Image sources (pluggable so tests + offline replays don't need hardware)
# ============================================================================


class ImageSource:
    """Returns a PIL.Image when called. Subclass for ZED / dir / static."""

    name: str = "image-source"

    def __call__(self) -> Image.Image:
        raise NotImplementedError

    def close(self) -> None:
        pass


class StaticImageSource(ImageSource):
    name = "static"

    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(path)
        self.img = Image.open(path).convert("RGB")
        print(f"[image] static image: {path} ({self.img.size})")

    def __call__(self) -> Image.Image:
        return self.img


class DirReplayImageSource(ImageSource):
    name = "dir"

    def __init__(self, dir_path: Path, fps: float = 1.0):
        self.frames = sorted(
            list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
        )
        if not self.frames:
            raise FileNotFoundError(f"no images in {dir_path}")
        self.idx = 0
        print(f"[image] replay dir: {dir_path} — {len(self.frames)} frames")

    def __call__(self) -> Image.Image:
        f = self.frames[self.idx % len(self.frames)]
        self.idx += 1
        return Image.open(f).convert("RGB")


class MJPEGImageSource(ImageSource):
    name = "mjpeg"

    def __init__(self, url: str):
        import urllib.request  # noqa: F401  (used in __call__)
        self.url = url
        # We'll grab one frame per call; stream-parsing the multipart MJPEG
        # is robust in cv2 but we keep this dep-free.
        print(f"[image] MJPEG stream: {url}")

    def __call__(self) -> Image.Image:
        import urllib.request, io
        with urllib.request.urlopen(self.url, timeout=2.0) as r:
            data = r.read()
        # Find a JPEG in the buffer (works if URL serves a single image
        # or an MJPEG multipart frame boundary).
        i = data.find(b"\xff\xd8")
        j = data.find(b"\xff\xd9", i)
        if i == -1 or j == -1:
            raise RuntimeError("no JPEG found in MJPEG payload")
        return Image.open(io.BytesIO(data[i:j + 2])).convert("RGB")


class ZEDImageSource(ImageSource):
    """Live ZED camera frames via the ZED SDK Python wrapper.

    This is intentionally lazy-imported and raises a clear error if the
    SDK isn't installed — we don't want the offline-replay paths to break
    just because pyzed isn't on this box.
    """
    name = "zed"

    def __init__(self, resolution: str = "VGA", fps: int = 30):
        try:
            import pyzed.sl as sl
        except ImportError as e:
            raise ImportError(
                "pyzed SDK not installed. Install the ZED SDK + python "
                "wrapper from https://www.stereolabs.com/developers/release "
                "or use --image-source dir for offline replay."
            ) from e
        self.sl = sl
        cam = sl.Camera()
        params = sl.InitParameters()
        params.camera_resolution = getattr(sl.RESOLUTION, resolution)
        params.camera_fps = fps
        params.coordinate_units = sl.UNIT.METER
        err = cam.open(params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {err}")
        self.cam = cam
        self.runtime = sl.RuntimeParameters()
        self.left = sl.Mat()
        print(f"[image] ZED open at {resolution}/{fps}fps")

    def __call__(self) -> Image.Image:
        if self.cam.grab(self.runtime) != self.sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED grab failed")
        self.cam.retrieve_image(self.left, self.sl.VIEW.LEFT)
        # ZED returns BGRA; drop alpha and BGR→RGB.
        bgra = self.left.get_data()
        rgb = bgra[..., [2, 1, 0]]
        return Image.fromarray(np.ascontiguousarray(rgb))

    def close(self) -> None:
        self.cam.close()


def make_image_source(args) -> ImageSource:
    src = args.image_source
    if src == "static":
        return StaticImageSource(Path(args.image_path))
    if src == "dir":
        return DirReplayImageSource(Path(args.image_dir))
    if src == "mjpeg":
        return MJPEGImageSource(args.mjpeg_url)
    if src == "zed":
        return ZEDImageSource(resolution=args.zed_resolution,
                               fps=args.zed_fps)
    raise SystemExit(f"unknown --image-source: {src}")


# ============================================================================
# Actuators
# ============================================================================


class Actuator:
    """Sends a (vx, vy, vyaw) command to the robot (or logs it)."""

    name: str = "actuator"

    def init(self) -> None:
        pass

    def send(self, vx: float, vy: float, vyaw: float) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass


class PrintActuator(Actuator):
    """No SDK, no robot. Just print the command."""
    name = "print"

    def __init__(self):
        self._last = None

    def send(self, vx, vy, vyaw):
        cmd = (round(vx, 3), round(vy, 3), round(vyaw, 3))
        if cmd != self._last:
            print(f"[actuator/print] would send Move({vx:+.3f}, "
                  f"{vy:+.3f}, {vyaw:+.3f})")
            self._last = cmd


class DryRunActuator(Actuator):
    """Initialise the SDK and connect, but never call Move()."""
    name = "dry"

    def __init__(self, net: str):
        self.net = net
        self._client = None

    def init(self) -> None:
        try:
            from booster_robotics_sdk_python import (
                B1LocoClient, ChannelFactory,
            )
        except ImportError as e:
            raise ImportError(
                "booster_robotics_sdk_python not installed. Either install "
                "it (per Booster docs) or use --mode print which needs no "
                "SDK at all."
            ) from e
        print(f"[actuator/dry] SDK ChannelFactory init on {self.net}")
        ChannelFactory.Instance().Init(0, self.net)
        self._client = B1LocoClient()
        self._client.Init()
        print("[actuator/dry] B1LocoClient initialised. NOT switching to "
              "kWalking. Move() calls will be logged, not sent.")

    def send(self, vx, vy, vyaw):
        # Intentionally do not call self._client.Move().
        # Log every command so the operator can verify NaVILA's outputs.
        print(f"[actuator/dry] would send Move({vx:+.3f}, {vy:+.3f}, "
              f"{vyaw:+.3f})  (NOT SENT — dry mode)")

    def shutdown(self) -> None:
        # Nothing to clean up — we never changed mode.
        pass


class LiveActuator(Actuator):
    """Real robot. Switches to kWalking on init, kDamping on exit, sends
    Move() at the requested rate."""
    name = "live"

    def __init__(self, net: str, send_hz: float = 20.0,
                 watchdog_seconds: float = 1.5):
        self.net = net
        self.send_hz = send_hz
        self.watchdog_seconds = watchdog_seconds
        self._client = None
        self._mode = None
        self._cmd = (0.0, 0.0, 0.0)
        self._cmd_lock = threading.Lock()
        self._cmd_set_at = 0.0
        self._abort = threading.Event()
        self._sender_thread: Optional[threading.Thread] = None

    def init(self) -> None:
        from booster_robotics_sdk_python import (
            B1LocoClient, ChannelFactory, RobotMode,
        )
        self._RobotMode = RobotMode
        print(f"[actuator/live] SDK ChannelFactory init on {self.net}")
        ChannelFactory.Instance().Init(0, self.net)
        self._client = B1LocoClient()
        self._client.Init()
        print("[actuator/live] switching to kWalking ...")
        self._client.ChangeMode(RobotMode.kWalking)
        self._mode = RobotMode.kWalking
        time.sleep(0.5)
        # Background sender ensures Move() is called at SEND_HZ even when
        # the planner only updates the command every ~400 ms (NaVILA
        # inference latency).  If the cmd hasn't been refreshed within
        # watchdog_seconds we send (0, 0, 0) — a stale command holding
        # last velocity could carry the robot into a wall after the
        # planner has moved on.
        self._cmd_set_at = time.perf_counter()
        self._sender_thread = threading.Thread(
            target=self._sender_loop, name="actuator-sender", daemon=True,
        )
        self._sender_thread.start()
        print("[actuator/live] sender thread started.")

    def send(self, vx, vy, vyaw):
        with self._cmd_lock:
            self._cmd = (float(vx), float(vy), float(vyaw))
            self._cmd_set_at = time.perf_counter()

    def shutdown(self) -> None:
        print("[actuator/live] shutdown: zero cmd, then kDamping ...")
        self._abort.set()
        if self._sender_thread is not None:
            self._sender_thread.join(timeout=2.0)
        # Try to zero the cmd, but ALWAYS attempt to dampen — leaving the
        # robot in kWalking on a crashed shutdown is the worst-case outcome.
        try:
            self._client.Move(0.0, 0.0, 0.0)
            time.sleep(0.2)
        except Exception as e:
            print(f"[actuator/live] zero Move() failed: {e!r}",
                  file=sys.stderr)
        try:
            self._client.ChangeMode(self._RobotMode.kDamping)
            print("[actuator/live] kDamping.")
        except Exception as e:
            print(f"[actuator/live] kDamping ChangeMode failed: {e!r}",
                  file=sys.stderr)

    def _sender_loop(self):
        period = 1.0 / self.send_hz
        next_tick = time.perf_counter()
        while not self._abort.is_set():
            with self._cmd_lock:
                vx, vy, vyaw = self._cmd
                age = time.perf_counter() - self._cmd_set_at
            if age > self.watchdog_seconds:
                vx = vy = vyaw = 0.0  # watchdog: stale cmd → stop
            try:
                self._client.Move(vx, vy, vyaw)
            except Exception as e:
                print(f"[actuator/live] Move() failed: {e!r}", file=sys.stderr)
            next_tick += period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.perf_counter()


def make_actuator(args) -> Actuator:
    if args.mode == "print":
        return PrintActuator()
    if args.mode == "dry":
        return DryRunActuator(net=args.net)
    if args.mode == "live":
        return LiveActuator(net=args.net, send_hz=args.send_hz,
                             watchdog_seconds=args.watchdog_seconds)
    raise SystemExit(f"unknown --mode: {args.mode}")


# ============================================================================
# Optional pose source (for heading-assist + proximity termination on real)
# ============================================================================


class PoseSource:
    """Returns (xy, yaw) or (None, None) if pose isn't available.

    Default: no pose. Subclass + plug in to enable heading-assist /
    proximity termination on the real robot. Possible implementations:
    - SDK low-state subscriber (yaw from IMU)
    - External SLAM / VICON
    - Visual odometry from the ZED itself
    """

    def read(self) -> tuple[Optional[tuple[float, float]], Optional[float]]:
        return None, None

    def close(self) -> None:
        pass


# ============================================================================
# Main loop
# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--instruction", required=True,
                     help="Multi-step instruction (split on |/;/then).")
    ap.add_argument("--model-path", type=Path, default=DEFAULT_CKPT)

    # mode + actuator
    ap.add_argument("--mode", choices=["print", "dry", "live"],
                     default="print",
                     help="print = log only, no SDK; dry = SDK init but no "
                          "Move(); live = real motion.")
    ap.add_argument("--net", type=str, default="127.0.0.1",
                     help="ChannelFactory network interface / robot IP.")
    ap.add_argument("--send-hz", type=float, default=20.0,
                     help="Move() send rate (live mode).")
    ap.add_argument("--watchdog-seconds", type=float, default=1.5,
                     help="Send (0,0,0) if the planner cmd is older than this.")

    # image source
    ap.add_argument("--image-source", choices=["zed", "mjpeg", "dir", "static"],
                     default="static")
    ap.add_argument("--image-path", type=Path,
                     default=Path.home() / "Projects/k1_research/experiments/vla/test_image.jpg",
                     help="--image-source static")
    ap.add_argument("--image-dir", type=Path,
                     help="--image-source dir")
    ap.add_argument("--mjpeg-url", type=str, default="http://localhost:8080/stream",
                     help="--image-source mjpeg")
    ap.add_argument("--zed-resolution", type=str, default="VGA",
                     help="--image-source zed (HD2K/HD1080/HD720/VGA)")
    ap.add_argument("--zed-fps", type=int, default=30)

    # planner
    ap.add_argument("--per-step-time", type=float, default=25.0)
    ap.add_argument("--proximity-threshold", type=float, default=1.0)
    ap.add_argument("--max-sim-seconds", type=float, default=120.0)
    ap.add_argument("--action-duration", type=float, default=ACTION_DURATION)

    # caps
    ap.add_argument("--vx-max", type=float, default=0.4,
                     help="Conservative on-robot caps; sim defaults are 0.6.")
    ap.add_argument("--vy-max", type=float, default=0.15)
    ap.add_argument("--vyaw-max", type=float, default=0.4)

    # controllers
    ap.add_argument("--heading-assist", action=argparse.BooleanOptionalAction,
                     default=False,
                     help="Requires a PoseSource. Off by default on the real "
                          "robot since most setups don't have odometry.")
    ap.add_argument("--heading-kp", type=float, default=1.5)
    ap.add_argument("--turn-controller", action=argparse.BooleanOptionalAction,
                     default=True,
                     help="Bypass NaVILA for pure-turn sub-steps. Requires a "
                          "PoseSource for the yaw delta — if no pose, turns "
                          "fall back to NaVILA-only execution.")
    ap.add_argument("--turn-kp", type=float, default=2.0)
    ap.add_argument("--turn-min-vyaw", type=float, default=0.30)
    ap.add_argument("--turn-tolerance-deg", type=float, default=5.0)

    # frame buffer
    ap.add_argument("--frame-buffer-period", type=float, default=0.4,
                     help="Push a head-camera frame into the VLM buffer every "
                          "this many seconds.")

    args = ap.parse_args()

    print(f"[main] mode={args.mode}  image_source={args.image_source}")
    print(f"[main] caps  vx<={args.vx_max} vy<={args.vy_max} "
          f"vyaw<={args.vyaw_max}")
    print(f"[main] instruction = {args.instruction!r}")

    # ----------------------------------------------------------------- plan
    substeps = parse_substeps(
        args.instruction, DEFAULT_SCENE_TARGETS,
        default_time=args.per_step_time,
        proximity_threshold=args.proximity_threshold,
    )
    print(f"[plan] {len(substeps)} sub-step(s):")
    for i, ss in enumerate(substeps):
        print("  " + describe_substep(i, len(substeps), ss))

    # ---------------------------------------------------------- io / vlm
    image_src = make_image_source(args)
    actuator = make_actuator(args)
    pose = PoseSource()  # no-op by default; subclass to enable assist

    vlm = VLMRunner(
        args.model_path, args.action_duration,
        args.vx_max, args.vy_max, args.vyaw_max,
    )
    vlm.load_model()
    bootstrap = image_src()
    vlm.bootstrap_buffer(bootstrap)
    vlm.set_instruction(substeps[0].instruction)
    vlm.start()

    # --------------------------------------------------------- actuator
    actuator.init()

    # ---------------------------------------------------------- run loop
    sim_t0 = time.perf_counter()
    next_frame_push = time.perf_counter()
    next_print = time.perf_counter()

    drain_deadline: Optional[float] = None
    state = TerminationState(
        step_idx=0,
        started_at=time.perf_counter(),
        start_yaw=0.0,
        last_yaw=0.0,
    )
    if pose.read()[1] is not None:
        _, y0 = pose.read()
        state.start_yaw = state.last_yaw = y0  # type: ignore

    print(f"[main] starting sub-step 1: {substeps[0].instruction!r}")
    if args.mode == "live":
        print("[main] LIVE mode — robot WILL move. Ctrl-C to stop.")

    try:
        while True:
            now = time.perf_counter()

            # 1) refresh pose (no-op unless caller plugged a real source)
            cur_xy, cur_yaw = pose.read()
            if cur_yaw is not None:
                update_yaw_unwrap(state, cur_yaw)

            # 2) get latest VLM cmd
            vlm_cmd = vlm.get_command()
            vlm_stop = vlm.stop_event.is_set()
            cur_ss = substeps[state.step_idx]

            # 3) check if this sub-step is done
            if drain_deadline is None:
                done_reason = check_termination(
                    cur_ss, state,
                    current_pos_xy=cur_xy,
                    vlm_stop=vlm_stop,
                    now=now,
                    yaw_tolerance_deg=args.turn_tolerance_deg,
                )
                if done_reason is not None:
                    print(f"[plan] sub-step {state.step_idx + 1}/{len(substeps)} "
                          f"done: {done_reason}", flush=True)
                    state.step_idx += 1
                    if state.step_idx >= len(substeps):
                        print("[plan] all sub-steps complete — sending zero "
                              "for 1.5 s then exiting.")
                        drain_deadline = now + 1.5
                    else:
                        next_ss = substeps[state.step_idx]
                        print(f"[plan] starting sub-step "
                              f"{state.step_idx + 1}/{len(substeps)}: "
                              f"{next_ss.instruction!r}")
                        state.started_at = now
                        if cur_yaw is not None:
                            state.start_yaw = state.last_yaw = cur_yaw
                        state.yaw_unwrap = 0.0
                        state.min_distance = float("inf")
                        vlm.clear_stop()
                        vlm.set_instruction(next_ss.instruction)
                        vlm_cmd = (0.0, 0.0, 0.0)

            # 4) controllers
            if drain_deadline is not None:
                out = type("Out", (), dict(vx=0.0, vy=0.0, vyaw=0.0,
                                            tag="DRAIN"))()
            else:
                out = apply_controllers(
                    ss=cur_ss,
                    state=state,
                    current_pos_xy=cur_xy,
                    current_yaw=cur_yaw,
                    vlm_cmd=vlm_cmd,
                    vx_max=args.vx_max,
                    vy_max=args.vy_max,
                    vyaw_max=args.vyaw_max,
                    heading_assist=args.heading_assist,
                    heading_kp=args.heading_kp,
                    turn_controller=args.turn_controller,
                    turn_kp=args.turn_kp,
                    turn_min_vyaw=args.turn_min_vyaw,
                )

            # 5) send to actuator
            actuator.send(out.vx, out.vy, out.vyaw)

            # 6) push a head frame into the VLM buffer at the configured rate
            if now >= next_frame_push:
                try:
                    img = image_src()
                    vlm.push_frame(img)
                except Exception as e:
                    print(f"[image] grab failed: {e!r}", file=sys.stderr)
                next_frame_push = now + args.frame_buffer_period

            # 7) periodic console line
            if now >= next_print:
                s = vlm.status()
                pose_str = (f"pose=(?, ?) yaw=?" if cur_xy is None
                            else f"pose=({cur_xy[0]:+.2f},{cur_xy[1]:+.2f}) "
                                  f"yaw={math.degrees(cur_yaw or 0.0):+.0f}°")
                print(f"[main t={now - sim_t0:5.1f}s] "
                      f"applied[{out.tag}] vx={out.vx:+.2f} vy={out.vy:+.2f} "
                      f"vyaw={out.vyaw:+.2f}  vlm({s['vx']:+.2f},{s['vy']:+.2f},"
                      f"{s['vyaw']:+.2f}) {pose_str} "
                      f"vlm_inf={s['inf_count']}", flush=True)
                next_print = now + 1.0

            # 8) terminate?
            if drain_deadline is not None and now >= drain_deadline:
                print("[main] drain complete.")
                break
            if (now - sim_t0) > args.max_sim_seconds:
                print("[main] hit --max-sim-seconds.")
                if drain_deadline is None:
                    drain_deadline = now + 1.5

            # 9) loop pace — main loop runs much faster than VLM; aim ~50 Hz
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[main] Ctrl-C — shutting down.")
    finally:
        print("[main] cleanup ...")
        actuator.shutdown()
        vlm.shutdown()
        image_src.close()
        pose.close()


if __name__ == "__main__":
    main()
