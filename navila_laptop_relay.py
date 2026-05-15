#!/usr/bin/env python3
"""Laptop relay: K1 SDK ↔ Tailscale ↔ desktop NaVILA server.

Runs on the lab laptop. The laptop has NO GPU and does NOT load NaVILA.
It only:

  1. Connects to the K1 via the Booster SDK (B1LocoClient over Ethernet).
  2. Connects to the desktop NaVILA server over Tailscale (TCP 5555).
  3. Grabs frames from the head camera (ZED / MJPEG / dir / static).
  4. Streams JPEG frames to the desktop, reads back (vx, vy, vyaw) + raw text.
  5. Calls B1LocoClient.Move(vx, vy, vyaw) on the K1.
  6. Shows a live HUD on screen so you can see what's happening while
     walking with the robot.

Modes (same progression as navila_k1_realrobot.py):

  ``--mode print``  no SDK at all; just print would-be Move() calls.
                    Use for end-to-end network smoke tests.
  ``--mode dry``    SDK init + connect, but Move() is logged not sent.
                    Use to validate the pipeline on the powered robot
                    before going live.
  ``--mode live``   send Move() at SEND_HZ. kWalking on enter,
                    kPrepare on exit (robot stays standing). Press ``q``
                    in the HUD window or hit Ctrl-C to emergency-stop.

Examples
========

End-to-end smoke test with a static image, no SDK, no robot::

    python navila_laptop_relay.py --mode print \\
        --image-source static \\
        --server <desktop-tailscale-ip> \\
        --instruction "walk to the red box"

Powered-robot dry-run (no motion) with ZED on the laptop::

    python navila_laptop_relay.py --mode dry \\
        --image-source zed --net 192.168.10.102 \\
        --server <desktop-tailscale-ip> \\
        --instruction "walk forward 3 meters | turn left 90 deg"

Live deploy::

    python navila_laptop_relay.py --mode live \\
        --image-source zed --net 192.168.10.102 \\
        --server <desktop-tailscale-ip> \\
        --instruction "walk to the chair | turn right 90 deg | walk forward"
"""

from __future__ import annotations

import argparse
import io
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import navila_protocol as proto  # noqa: E402
from navila_k1_realrobot import (  # noqa: E402
    DryRunActuator, LiveActuator, PrintActuator,
    DirReplayImageSource, MJPEGImageSource, StaticImageSource, ZEDImageSource,
)
from navila_k1_bridge import ACTION_DURATION  # noqa: E402


# ============================================================================
# Remote VLM client — owns the TCP socket to the desktop server
# ============================================================================


@dataclass
class RemoteState:
    """The latest state snapshot we got back from the server."""
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    tag: str = "INIT"
    label: str = "(waiting for first server response)"
    raw: str = ""
    step_idx: int = 0
    step_total: int = 0
    step_instruction: str = ""
    done_reason: Optional[str] = None
    all_done: bool = False
    drain_done: bool = False
    vlm_stop: bool = False
    inf_count: int = 0
    inf_ms: float = 0.0
    buffer_size: int = 0
    last_update_at: float = 0.0
    # Network status (set by the tick thread, not the server)
    connection_lost: bool = False
    last_error: Optional[str] = None


class RemoteVLMClient:
    """Maintains the TCP link to the server and a background tick loop.

    The tick loop pushes the most recently captured frame plus optional
    pose into a ``tick`` message at ``period`` seconds, and stores the
    server's response in ``self.state``. Main thread reads ``state``
    via :meth:`get_state` (snapshot, no lock kept).
    """

    def __init__(self, host: str, port: int, instruction: str,
                 cfg: dict, period: float, io_timeout: float = 5.0):
        self.host = host
        self.port = port
        self.instruction = instruction
        self.cfg = cfg
        self.period = period
        self.io_timeout = io_timeout

        self.sock: Optional[socket.socket] = None
        self._state_lock = threading.Lock()
        self._state = RemoteState()

        # Latest frame to send on next tick. Tick thread consumes it.
        self._frame_lock = threading.Lock()
        self._next_jpeg: Optional[bytes] = None

        # Latest pose to send on next tick.
        self._pose_lock = threading.Lock()
        self._pose_xy: Optional[tuple[float, float]] = None
        self._pose_yaw: Optional[float] = None

        self._abort = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.step_count = 0
        self.steps: list[dict] = []

    # ---------------------------------------------------- connect / handshake

    def connect_and_handshake(self) -> None:
        print(f"[remote] connecting to {self.host}:{self.port} "
              f"(timeout={self.io_timeout}s) ...", flush=True)
        self.sock = proto.connect(self.host, self.port,
                                   connect_timeout=10.0,
                                   io_timeout=self.io_timeout)
        proto.send_msg(self.sock, {"type": "hello",
                                    "client": "laptop-relay",
                                    "version": 1})
        ack, _ = proto.recv_msg(self.sock)
        if ack.get("type") != "hello_ack":
            raise RuntimeError(f"unexpected handshake: {ack!r}")
        print(f"[remote] handshake ok — server={ack.get('server')!r} "
              f"model_loaded={ack.get('model_loaded')}", flush=True)

        msg = {"type": "set_instruction", "instruction": self.instruction}
        msg.update(self.cfg)
        proto.send_msg(self.sock, msg)
        ack, _ = proto.recv_msg(self.sock)
        if ack.get("type") != "instruction_ack" or not ack.get("ok"):
            raise RuntimeError(f"set_instruction failed: {ack!r}")
        self.step_count = int(ack.get("step_count", 0))
        self.steps = ack.get("steps", [])
        print(f"[remote] instruction set — {self.step_count} sub-step(s):",
              flush=True)
        for i, s in enumerate(self.steps):
            extra = []
            if s.get("yaw_target_deg") is not None:
                extra.append(f"yaw={s['yaw_target_deg']:+.0f}°")
            if s.get("proximity_target") is not None:
                p = s["proximity_target"]
                extra.append(f"prox=({p[0]:+.1f},{p[1]:+.1f})")
            extra.append(f"tlim={s['time_limit']:.0f}s")
            print(f"  step {i + 1}/{self.step_count}: {s['instruction']!r}  "
                  + "  ".join(extra))

    # ---------------------------------------------------- producer side (main)

    def set_frame_jpeg(self, jpeg: bytes) -> None:
        """Stage a JPEG to be sent on the next tick. Main thread calls this."""
        with self._frame_lock:
            self._next_jpeg = jpeg

    def set_pose(self, xy: Optional[tuple[float, float]],
                  yaw: Optional[float]) -> None:
        with self._pose_lock:
            self._pose_xy = xy
            self._pose_yaw = yaw

    def get_state(self) -> RemoteState:
        with self._state_lock:
            # Return a shallow copy so main thread can read without holding lock
            return RemoteState(**self._state.__dict__)

    # ---------------------------------------------------- tick loop (thread)

    def start_ticking(self) -> None:
        self._thread = threading.Thread(target=self._tick_loop,
                                          name="remote-tick", daemon=True)
        self._thread.start()

    def _tick_loop(self) -> None:
        next_tick = time.perf_counter()
        while not self._abort.is_set():
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.05))
                continue

            with self._frame_lock:
                jpeg = self._next_jpeg
                self._next_jpeg = None
            with self._pose_lock:
                pose_xy = self._pose_xy
                pose_yaw = self._pose_yaw

            header = {
                "type": "tick",
                "have_image": jpeg is not None,
                "have_pose": pose_xy is not None or pose_yaw is not None,
                "pose_xy": list(pose_xy) if pose_xy is not None else None,
                "pose_yaw": pose_yaw,
            }
            try:
                proto.send_msg(self.sock, header, blob=jpeg or b"")
                resp, _ = proto.recv_msg(self.sock)
            except (ConnectionError, socket.timeout, OSError) as e:
                with self._state_lock:
                    self._state.connection_lost = True
                    self._state.last_error = repr(e)
                    self._state.vx = self._state.vy = self._state.vyaw = 0.0
                print(f"[remote] network error: {e!r} — tick thread exits",
                      flush=True)
                return

            if resp.get("type") == "state":
                with self._state_lock:
                    self._state.vx = float(resp["vx"])
                    self._state.vy = float(resp["vy"])
                    self._state.vyaw = float(resp["vyaw"])
                    self._state.tag = resp.get("tag", "")
                    self._state.label = resp.get("label", "")
                    self._state.raw = resp.get("raw", "")
                    self._state.step_idx = int(resp.get("step_idx", 0))
                    self._state.step_total = int(resp.get("step_total", 0))
                    self._state.step_instruction = resp.get(
                        "step_instruction", "")
                    self._state.done_reason = resp.get("done_reason")
                    self._state.all_done = bool(resp.get("all_done"))
                    self._state.drain_done = bool(resp.get("drain_done"))
                    self._state.vlm_stop = bool(resp.get("vlm_stop"))
                    self._state.inf_count = int(resp.get("inf_count", 0))
                    self._state.inf_ms = float(resp.get("inf_ms", 0.0))
                    self._state.buffer_size = int(resp.get("buffer_size", 0))
                    self._state.last_update_at = time.perf_counter()
                if resp.get("step_advanced"):
                    print(f"[remote] sub-step advanced → "
                          f"{resp.get('step_idx')}/{resp.get('step_total')}  "
                          f"reason={resp.get('done_reason')!r}", flush=True)
            elif resp.get("type") == "error":
                with self._state_lock:
                    self._state.last_error = resp.get("message", "")
                print(f"[remote] server error: {resp.get('message')}",
                      flush=True)

            next_tick += self.period

    # ---------------------------------------------------- shutdown

    def shutdown(self) -> None:
        self._abort.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.sock is not None:
            try:
                proto.send_msg(self.sock, {"type": "shutdown"})
                # We don't wait for ack — already shutting down.
            except Exception:
                pass
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()
            self.sock = None


# ============================================================================
# Image source factory (reuses the existing PIL-returning classes)
# ============================================================================


def make_image_source(args):
    src = args.image_source
    if src == "static":
        return StaticImageSource(Path(args.image_path))
    if src == "dir":
        return DirReplayImageSource(Path(args.image_dir))
    if src == "mjpeg":
        return MJPEGImageSource(args.mjpeg_url)
    if src == "zed":
        return ZEDImageSource(resolution=args.zed_resolution, fps=args.zed_fps)
    raise SystemExit(f"unknown --image-source: {src}")


def make_actuator(args):
    if args.mode == "print":
        return PrintActuator()
    if args.mode == "dry":
        return DryRunActuator(net=args.net)
    if args.mode == "live":
        return LiveActuator(net=args.net, send_hz=args.send_hz,
                             watchdog_seconds=args.watchdog_seconds)
    raise SystemExit(f"unknown --mode: {args.mode}")


def encode_jpeg(pil_img: Image.Image, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ============================================================================
# HUD overlay
# ============================================================================


def draw_hud(pil_img: Image.Image, state: RemoteState, mode: str,
             instruction: str, last_grab_ms: float, last_send_age_s: float,
             out_vx: float, out_vy: float, out_vyaw: float):
    """Compose a BGR numpy image with a HUD overlay and return it.

    Returns ``None`` if cv2 isn't available; caller should fall back to
    a console line in that case.
    """
    try:
        import cv2
    except ImportError:
        return None

    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    # Top banner.
    cv2.rectangle(bgr, (0, 0), (w, 86), (0, 0, 0), -1)
    cv2.putText(bgr, f"[{mode.upper()}] {instruction}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    step_str = (f"step {state.step_idx + 1}/{state.step_total}"
                if state.step_total > 0 else "step -/-")
    if state.all_done:
        step_str = f"plan complete ({state.step_total}/{state.step_total})"
    cv2.putText(bgr, f"{step_str}: {state.step_instruction}",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1,
                cv2.LINE_AA)
    raw_line = (state.raw[:80] + "...") if len(state.raw) > 80 else state.raw
    cv2.putText(bgr, f"NaVILA: {raw_line!r}",
                (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1,
                cv2.LINE_AA)

    # Bottom banner.
    cv2.rectangle(bgr, (0, h - 76), (w, h), (0, 0, 0), -1)
    cv2.putText(bgr,
                f"out: vx={out_vx:+.2f} vy={out_vy:+.2f} vyaw={out_vyaw:+.2f} "
                f"[{state.tag}]",
                (8, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bgr,
                f"vlm: inf#{state.inf_count} {state.inf_ms:5.0f}ms  "
                f"buf={state.buffer_size}  "
                f"stop={state.vlm_stop}  "
                f"link_age={last_send_age_s:.2f}s  "
                f"grab={last_grab_ms:.0f}ms",
                (8, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(bgr, "press 'q' to emergency-stop",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (120, 120, 255), 1, cv2.LINE_AA)

    if state.connection_lost:
        cv2.putText(bgr, "!! NETWORK DOWN !!",
                    (w // 2 - 140, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA)
    if state.all_done:
        cv2.putText(bgr, "PLAN COMPLETE",
                    (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2, cv2.LINE_AA)

    return bgr


def _clip(v: float, cap: float) -> float:
    return max(-cap, min(cap, v))


# ============================================================================
# Main loop
# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instruction", required=True,
                     help="Multi-step instruction (split on |/;/then).")

    # remote server
    ap.add_argument("--server", required=True,
                     help="Desktop server host/IP (Tailscale).")
    ap.add_argument("--port", type=int, default=proto.DEFAULT_PORT)
    ap.add_argument("--io-timeout", type=float, default=5.0,
                     help="Socket read timeout (s). Hits the watchdog if "
                          "the server stops responding.")

    # mode + actuator
    ap.add_argument("--mode", choices=["print", "dry", "live"],
                     default="print")
    ap.add_argument("--net", type=str, default="192.168.10.102",
                     help="K1 IP / Ethernet interface for ChannelFactory.")
    ap.add_argument("--send-hz", type=float, default=20.0,
                     help="Move() send rate inside LiveActuator.")
    ap.add_argument("--watchdog-seconds", type=float, default=1.5,
                     help="LiveActuator zeroes cmd if planner update older.")

    # image source
    ap.add_argument("--image-source",
                     choices=["zed", "mjpeg", "dir", "static"],
                     default="static")
    ap.add_argument("--image-path", type=Path,
                     default=Path.home() / "Projects/k1_research/experiments/vla/test_image.jpg")
    ap.add_argument("--image-dir", type=Path)
    ap.add_argument("--mjpeg-url", type=str,
                     default="http://192.168.10.102:8080/stream",
                     help="MJPEG stream (use this if the ZED is plugged "
                          "into the K1 onboard PC, not the laptop).")
    ap.add_argument("--zed-resolution", default="VGA")
    ap.add_argument("--zed-fps", type=int, default=30)
    ap.add_argument("--jpeg-quality", type=int, default=80,
                     help="JPEG quality (1-100) for frames sent to server.")

    # planner config (forwarded to server in set_instruction)
    ap.add_argument("--per-step-time", type=float, default=25.0)
    ap.add_argument("--proximity-threshold", type=float, default=1.0)
    ap.add_argument("--action-duration", type=float, default=ACTION_DURATION)
    ap.add_argument("--vx-max", type=float, default=0.4)
    ap.add_argument("--vy-max", type=float, default=0.15)
    ap.add_argument("--vyaw-max", type=float, default=0.4)
    ap.add_argument("--heading-assist",
                     action=argparse.BooleanOptionalAction, default=False,
                     help="Server-side; needs pose. Off by default.")
    ap.add_argument("--turn-controller",
                     action=argparse.BooleanOptionalAction, default=False,
                     help="Server-side; needs pose. Off by default — without "
                          "pose, this would open-loop spin until time-out.")

    # cadence
    ap.add_argument("--tick-period", type=float, default=0.4,
                     help="How often to send a frame+poll to the server. "
                          "NaVILA runs at ~1Hz so 0.4s leaves room.")
    ap.add_argument("--display-hz", type=float, default=20.0,
                     help="HUD refresh rate.")
    ap.add_argument("--max-seconds", type=float, default=300.0,
                     help="Hard time limit; stops everything when reached.")
    ap.add_argument("--no-display", action="store_true",
                     help="Disable cv2 window (e.g. headless smoke test).")

    args = ap.parse_args()

    try:
        import cv2
    except ImportError:
        cv2 = None
        if not args.no_display:
            print("[laptop] cv2 not available — running with --no-display.",
                  file=sys.stderr)
            args.no_display = True

    # --- planner config sent to server in set_instruction ----------------
    cfg = {
        "per_step_time": args.per_step_time,
        "action_duration": args.action_duration,
        "vx_max": args.vx_max,
        "vy_max": args.vy_max,
        "vyaw_max": args.vyaw_max,
        "heading_assist": args.heading_assist,
        "turn_controller": args.turn_controller,
        "proximity_threshold": args.proximity_threshold,
    }

    # --- image source --------------------------------------------------------
    image_src = make_image_source(args)

    # --- remote --------------------------------------------------------------
    remote = RemoteVLMClient(host=args.server, port=args.port,
                              instruction=args.instruction, cfg=cfg,
                              period=args.tick_period,
                              io_timeout=args.io_timeout)
    remote.connect_and_handshake()

    # Bootstrap: grab a first frame *before* starting the tick loop so the
    # server has something in its buffer immediately.
    t_grab0 = time.perf_counter()
    first_frame = image_src()
    last_grab_ms = (time.perf_counter() - t_grab0) * 1000
    remote.set_frame_jpeg(encode_jpeg(first_frame, args.jpeg_quality))

    # --- actuator ------------------------------------------------------------
    actuator = make_actuator(args)
    print(f"[laptop] mode={args.mode}  image_source={args.image_source}  "
          f"server={args.server}:{args.port}", flush=True)
    print(f"[laptop] caps vx<={args.vx_max} vy<={args.vy_max} "
          f"vyaw<={args.vyaw_max}", flush=True)
    print(f"[laptop] instruction = {args.instruction!r}", flush=True)
    actuator.init()
    if args.mode == "live":
        print("[laptop] LIVE mode — K1 IS WALKING. press 'q' or Ctrl-C "
              "to emergency-stop.", flush=True)

    remote.start_ticking()

    # --- Ctrl-C → set a flag so the finally block runs cleanly --------------
    stop_requested = threading.Event()

    def _sigint(_signum, _frame):
        if not stop_requested.is_set():
            print("\n[laptop] SIGINT — emergency stopping ...", flush=True)
        stop_requested.set()

    signal.signal(signal.SIGINT, _sigint)

    # --- main loop -----------------------------------------------------------
    t0 = time.perf_counter()
    next_grab = time.perf_counter()
    next_print = time.perf_counter() + 1.0
    last_frame = first_frame
    exit_reason = "normal"

    try:
        while not stop_requested.is_set():
            now = time.perf_counter()

            # 1) Grab a new frame at the tick-period rate.
            if now >= next_grab:
                t_grab = time.perf_counter()
                try:
                    last_frame = image_src()
                    last_grab_ms = (time.perf_counter() - t_grab) * 1000
                    remote.set_frame_jpeg(encode_jpeg(last_frame,
                                                       args.jpeg_quality))
                except Exception as e:
                    print(f"[laptop] image grab failed: {e!r}",
                          file=sys.stderr)
                next_grab = now + args.tick_period

            # 2) Latest cmd from server.
            state = remote.get_state()
            if state.connection_lost:
                exit_reason = f"network down ({state.last_error})"
                break

            # 3) Defense-in-depth re-clip before send.
            vx = _clip(state.vx, args.vx_max)
            vy = _clip(state.vy, args.vy_max)
            vyaw = _clip(state.vyaw, args.vyaw_max)
            actuator.send(vx, vy, vyaw)

            # 4) Plan complete?
            if state.all_done and state.drain_done:
                exit_reason = "plan complete"
                break

            # 5) Time limit?
            if (now - t0) > args.max_seconds:
                exit_reason = f"max-seconds {args.max_seconds:.0f}s"
                break

            # 6) HUD.
            link_age = (now - state.last_update_at
                        if state.last_update_at > 0 else float("nan"))
            if not args.no_display and cv2 is not None:
                bgr = draw_hud(last_frame, state, args.mode, args.instruction,
                                last_grab_ms, link_age, vx, vy, vyaw)
                if bgr is not None:
                    cv2.imshow("NaVILA Laptop Relay", bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    exit_reason = "q-key emergency stop"
                    break

            # 7) Periodic console line (HUD-less fallback / log).
            if now >= next_print:
                print(f"[laptop t={now - t0:5.1f}s] "
                      f"step {state.step_idx + 1}/{state.step_total}  "
                      f"out[{state.tag}] vx={vx:+.2f} vy={vy:+.2f} "
                      f"vyaw={vyaw:+.2f}  "
                      f"raw={state.raw!r}  link_age={link_age:.2f}s  "
                      f"inf#{state.inf_count}",
                      flush=True)
                next_print = now + 1.0

            # 8) Loop pace — 1/display_hz feels snappy.
            sleep_s = max(0.005, 1.0 / max(args.display_hz, 1.0))
            time.sleep(sleep_s)

    except KeyboardInterrupt:
        # Belt-and-suspenders; our SIGINT handler should usually run first.
        exit_reason = "KeyboardInterrupt"
    except Exception as e:
        exit_reason = f"crash: {e!r}"
        import traceback
        traceback.print_exc()
    finally:
        print(f"[laptop] exit: {exit_reason}", flush=True)
        # ALWAYS try to zero the cmd first, even if shutdown will too.
        try:
            actuator.send(0.0, 0.0, 0.0)
        except Exception:
            pass
        try:
            actuator.shutdown()
        except Exception as e:
            print(f"[laptop] actuator.shutdown failed: {e!r}",
                  file=sys.stderr)
        try:
            remote.shutdown()
        except Exception:
            pass
        try:
            image_src.close()
        except Exception:
            pass
        if cv2 is not None and not args.no_display:
            cv2.destroyAllWindows()
        print("[laptop] bye.", flush=True)


if __name__ == "__main__":
    main()
