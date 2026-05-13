#!/usr/bin/env python3
"""Desktop NaVILA server — runs the GPU brain for the laptop relay.

Architecture:

    [laptop relay]  --TCP/Tailscale-->  [this server]
       |                                   |
       | (JPEG frames + tick)              | NaVILA inference
       | <-- (vx,vy,vyaw + raw text) ---   | Multi-step planner
                                           | 8-frame rolling buffer

Single-session: one laptop at a time. A new ``set_instruction`` resets
the rolling buffer, the planner state, and the VLM stop event.

This server owns:
  - NaVILA model (loaded once at startup; held across reconnects)
  - 8-frame rolling buffer (per session)
  - Multi-step planner state (sub-step idx, yaw_unwrap, timers)
  - The VLMRunner inference thread

The laptop owns:
  - The ZED camera grab
  - The K1 SDK (B1LocoClient.Move)
  - Wall-clock display

Run:

    python navila_server.py --bind 0.0.0.0 --port 5555 \\
        --model-path ~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f

The bind address ``0.0.0.0`` lets the laptop connect over the Tailscale
interface; firewall-restrict the port to the Tailscale subnet if your
LAN isn't already private.
"""

from __future__ import annotations

import argparse
import io
import math
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

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
)
from navila_k1_bridge import ACTION_DURATION  # noqa: E402
import navila_protocol as proto  # noqa: E402


DEFAULT_CKPT = Path.home() / "Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f"


# ============================================================================
# Per-session state (one connected laptop)
# ============================================================================


@dataclass
class SessionConfig:
    """Planner config sent by the laptop in ``set_instruction``."""
    instruction: str
    per_step_time: float = 25.0
    action_duration: float = ACTION_DURATION
    vx_max: float = 0.4
    vy_max: float = 0.15
    vyaw_max: float = 0.4
    heading_assist: bool = False
    turn_controller: bool = False  # off by default — needs pose
    heading_kp: float = 1.5
    turn_kp: float = 2.0
    turn_min_vyaw: float = 0.30
    turn_tolerance_deg: float = 5.0
    proximity_threshold: float = 1.0
    drain_seconds: float = 1.5


class Session:
    """One laptop's planner state, layered on top of a shared VLMRunner.

    The VLMRunner is shared across sessions (model is heavy); the
    rolling frame buffer, the instruction, and the sub-step idx are
    per-session. We achieve "per-session buffer" by resetting the
    VLMRunner's buffer at the start of each session — server accepts
    only one client at a time so there's no contention.
    """

    def __init__(self, vlm: VLMRunner, cfg: SessionConfig):
        self.vlm = vlm
        self.cfg = cfg

        # Apply caps the VLMRunner uses to clip parsed NaVILA actions.
        self.vlm.vx_max = cfg.vx_max
        self.vlm.vy_max = cfg.vy_max
        self.vlm.vyaw_max = cfg.vyaw_max
        self.vlm.action_duration = cfg.action_duration

        self.substeps: list[SubStep] = parse_substeps(
            cfg.instruction,
            DEFAULT_SCENE_TARGETS,
            default_time=cfg.per_step_time,
            proximity_threshold=cfg.proximity_threshold,
        )
        if not self.substeps:
            raise ValueError("instruction produced zero sub-steps")

        now = time.perf_counter()
        self.step_idx = 0
        self.state = TerminationState(
            step_idx=0, started_at=now, start_yaw=0.0, last_yaw=0.0,
        )
        self.all_done = False
        self.drain_deadline: Optional[float] = None
        self.last_done_reason: Optional[str] = None

        # Reset VLMRunner buffer + instruction for this session.
        self.vlm.clear_stop()
        with self.vlm._frame_lock:  # noqa: SLF001 — internal but stable
            self.vlm._frame_buffer.clear()  # noqa: SLF001
        self.vlm.set_instruction(self.substeps[0].instruction)

    # ----------------------------------------------------------- frames

    def push_jpeg(self, jpeg: bytes) -> int:
        """Decode JPEG → PIL → push into VLMRunner buffer. Return new size."""
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        with self.vlm._frame_lock:  # noqa: SLF001
            empty = len(self.vlm._frame_buffer) == 0  # noqa: SLF001
        if empty:
            # First frame: bootstrap all 8 slots so NaVILA can start
            # producing actions immediately (matches what the in-process
            # entry points do).
            self.vlm.bootstrap_buffer(img)
        else:
            self.vlm.push_frame(img)
        with self.vlm._frame_lock:  # noqa: SLF001
            return len(self.vlm._frame_buffer)  # noqa: SLF001

    # ----------------------------------------------------------- tick

    def tick(self, pose_xy: Optional[tuple[float, float]],
             pose_yaw: Optional[float]) -> dict:
        """Advance the planner one tick, return a state dict for the laptop.

        The laptop calls this at its main-loop rate (~10–20 Hz). Cheap
        ops only — NaVILA inference runs in its own thread inside the
        VLMRunner.
        """
        now = time.perf_counter()

        if pose_yaw is not None:
            update_yaw_unwrap(self.state, pose_yaw)

        vlm_cmd = self.vlm.get_command()
        vlm_stop = self.vlm.stop_event.is_set()
        # After the plan finishes step_idx == len(substeps); clamp so we
        # don't index past the end while draining.
        cur_ss = self.substeps[min(self.step_idx, len(self.substeps) - 1)]

        step_advanced = False
        done_reason: Optional[str] = None
        if self.drain_deadline is None and not self.all_done:
            done_reason = check_termination(
                cur_ss, self.state,
                current_pos_xy=pose_xy,
                vlm_stop=vlm_stop,
                now=now,
                yaw_tolerance_deg=self.cfg.turn_tolerance_deg,
            )
            if done_reason is not None:
                self.last_done_reason = done_reason
                self.step_idx += 1
                step_advanced = True
                if self.step_idx >= len(self.substeps):
                    self.all_done = True
                    self.drain_deadline = now + self.cfg.drain_seconds
                    cur_ss = self.substeps[-1]
                else:
                    next_ss = self.substeps[self.step_idx]
                    self.state.started_at = now
                    self.state.yaw_unwrap = 0.0
                    self.state.min_distance = float("inf")
                    if pose_yaw is not None:
                        self.state.start_yaw = pose_yaw
                        self.state.last_yaw = pose_yaw
                    self.vlm.clear_stop()
                    self.vlm.set_instruction(next_ss.instruction)
                    vlm_cmd = (0.0, 0.0, 0.0)
                    cur_ss = next_ss

        if self.drain_deadline is not None:
            vx = vy = vyaw = 0.0
            tag = "DRAIN"
        else:
            out = apply_controllers(
                ss=cur_ss,
                state=self.state,
                current_pos_xy=pose_xy,
                current_yaw=pose_yaw,
                vlm_cmd=vlm_cmd,
                vx_max=self.cfg.vx_max,
                vy_max=self.cfg.vy_max,
                vyaw_max=self.cfg.vyaw_max,
                heading_assist=self.cfg.heading_assist,
                heading_kp=self.cfg.heading_kp,
                turn_controller=self.cfg.turn_controller,
                turn_kp=self.cfg.turn_kp,
                turn_min_vyaw=self.cfg.turn_min_vyaw,
            )
            vx, vy, vyaw, tag = out.vx, out.vy, out.vyaw, out.tag

        vlm_status = self.vlm.status()
        with self.vlm._frame_lock:  # noqa: SLF001
            buf_size = len(self.vlm._frame_buffer)  # noqa: SLF001

        drain_done = (self.drain_deadline is not None
                      and now >= self.drain_deadline)
        return {
            "type": "state",
            "vx": float(vx), "vy": float(vy), "vyaw": float(vyaw),
            "tag": tag,
            "label": vlm_status["label"],
            "raw": vlm_status["raw"],
            "step_idx": self.step_idx if not self.all_done
                        else len(self.substeps),
            "step_total": len(self.substeps),
            "step_instruction": cur_ss.instruction,
            "step_advanced": step_advanced,
            "done_reason": done_reason,
            "all_done": bool(self.all_done),
            "drain_done": bool(drain_done),
            "vlm_stop": bool(vlm_stop),
            "inf_count": int(vlm_status["inf_count"]),
            "inf_ms": float(vlm_status["inf_ms"]),
            "buffer_size": int(buf_size),
            "ts": now,
        }


def substep_summary(ss: SubStep) -> dict:
    return {
        "instruction": ss.instruction,
        "yaw_target_deg": (math.degrees(ss.yaw_delta_target)
                           if ss.yaw_delta_target is not None else None),
        "proximity_target": (list(ss.proximity_target)
                             if ss.proximity_target is not None else None),
        "time_limit": float(ss.time_limit),
    }


# ============================================================================
# Connection handler
# ============================================================================


def handle_client(conn: socket.socket, addr: tuple, vlm: VLMRunner) -> None:
    """Single-client request loop. Returns on disconnect / error."""
    print(f"[server] client connected from {addr}")
    session: Optional[Session] = None

    try:
        # Initial handshake — optional but lets the client confirm we're up.
        header, _ = proto.recv_msg(conn)
        if header.get("type") != "hello":
            proto.send_msg(conn, {"type": "error",
                                   "message": f"expected hello, got {header.get('type')!r}"})
            return
        client_version = int(header.get("version", 0))
        print(f"[server]   hello from client={header.get('client')!r} "
              f"v{client_version}")
        proto.send_msg(conn, {"type": "hello_ack",
                               "server": "navila-server",
                               "model_loaded": True,
                               "version": 1})

        while True:
            try:
                header, blob = proto.recv_msg(conn)
            except (ConnectionError, socket.timeout) as e:
                print(f"[server]   socket error: {e!r} — closing")
                return

            mtype = header.get("type")

            if mtype == "set_instruction":
                cfg = SessionConfig(
                    instruction=header["instruction"],
                    per_step_time=float(header.get("per_step_time", 25.0)),
                    action_duration=float(header.get("action_duration",
                                                       ACTION_DURATION)),
                    vx_max=float(header.get("vx_max", 0.4)),
                    vy_max=float(header.get("vy_max", 0.15)),
                    vyaw_max=float(header.get("vyaw_max", 0.4)),
                    heading_assist=bool(header.get("heading_assist", False)),
                    turn_controller=bool(header.get("turn_controller", False)),
                    heading_kp=float(header.get("heading_kp", 1.5)),
                    turn_kp=float(header.get("turn_kp", 2.0)),
                    turn_min_vyaw=float(header.get("turn_min_vyaw", 0.30)),
                    turn_tolerance_deg=float(header.get("turn_tolerance_deg",
                                                          5.0)),
                    proximity_threshold=float(header.get("proximity_threshold",
                                                           1.0)),
                    drain_seconds=float(header.get("drain_seconds", 1.5)),
                )
                try:
                    session = Session(vlm, cfg)
                except Exception as e:
                    proto.send_msg(conn, {"type": "error",
                                           "message": f"set_instruction failed: {e!r}"})
                    continue
                print(f"[server]   set_instruction "
                      f"{cfg.instruction!r}  →  {len(session.substeps)} sub-step(s)")
                for i, ss in enumerate(session.substeps):
                    print("           " + describe_substep(
                        i, len(session.substeps), ss))
                proto.send_msg(conn, {
                    "type": "instruction_ack",
                    "ok": True,
                    "step_count": len(session.substeps),
                    "steps": [substep_summary(ss) for ss in session.substeps],
                })

            elif mtype == "tick":
                if session is None:
                    proto.send_msg(conn, {"type": "error",
                                           "message": "tick before set_instruction"})
                    continue
                if header.get("have_image") and blob:
                    try:
                        session.push_jpeg(blob)
                    except Exception as e:
                        print(f"[server]   JPEG decode failed: {e!r}")
                pose_xy = None
                pose_yaw = None
                if header.get("have_pose"):
                    pxy = header.get("pose_xy")
                    if pxy is not None and len(pxy) == 2:
                        pose_xy = (float(pxy[0]), float(pxy[1]))
                    py = header.get("pose_yaw")
                    if py is not None:
                        pose_yaw = float(py)
                state_msg = session.tick(pose_xy, pose_yaw)
                proto.send_msg(conn, state_msg)
                if state_msg.get("step_advanced"):
                    print(f"[server]   step advanced → "
                          f"{state_msg['step_idx']}/"
                          f"{state_msg['step_total']} "
                          f"reason={state_msg['done_reason']!r}")
                if state_msg.get("drain_done"):
                    print("[server]   drain complete — plan finished.")

            elif mtype == "shutdown":
                print("[server]   client requested shutdown")
                proto.send_msg(conn, {"type": "shutdown_ack"})
                return

            else:
                proto.send_msg(conn, {"type": "error",
                                       "message": f"unknown type: {mtype!r}"})

    except ConnectionError as e:
        print(f"[server]   client gone: {e!r}")
    except Exception:
        print("[server]   handler crashed:")
        traceback.print_exc()
    finally:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        conn.close()
        # Zero out cmd + stop VLM stop event so a stale state doesn't
        # leak into the next session.
        if session is not None:
            session.vlm.clear_stop()
        print(f"[server] client {addr} disconnected")


# ============================================================================
# Entry point
# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="0.0.0.0",
                     help="Bind interface. Default: all interfaces. "
                          "Use a Tailscale IP if you want to restrict.")
    ap.add_argument("--port", type=int, default=proto.DEFAULT_PORT)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_CKPT,
                     help="NaVILA checkpoint directory.")
    ap.add_argument("--io-timeout", type=float, default=60.0,
                     help="Per-connection socket read timeout (s).")
    args = ap.parse_args()

    print(f"[server] loading NaVILA from {args.model_path} (this is slow)...",
          flush=True)
    # Initialise the VLMRunner ONCE — model stays loaded across reconnects.
    vlm = VLMRunner(args.model_path)
    vlm.load_model()
    vlm.start()
    print("[server] model loaded and inference thread running.", flush=True)

    listener = proto.listen(args.bind, args.port, backlog=1)
    print(f"[server] listening on {args.bind}:{args.port} — Ctrl-C to exit.",
          flush=True)

    try:
        while True:
            conn, addr = listener.accept()
            conn.settimeout(args.io_timeout)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # Single-session: handle inline, don't accept a second client
            # until this one disconnects. Keeps the planner state
            # unambiguous and avoids concurrent VLM buffer writes.
            handle_client(conn, addr, vlm)
    except KeyboardInterrupt:
        print("\n[server] Ctrl-C — shutting down.")
    finally:
        listener.close()
        vlm.shutdown(timeout=2.0)
        print("[server] bye.")


if __name__ == "__main__":
    main()
