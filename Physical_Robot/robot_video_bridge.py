#!/usr/bin/env python3
"""On-robot ROS2 -> HTTP MJPEG bridge.

Runs on the K1's onboard PC. Subscribes to a ROS2 image topic (default
``/booster_video_stream``), keeps the latest frame as JPEG bytes, and serves
it over HTTP so a non-ROS2 host (the lab laptop on Ubuntu 24.04) can pull
frames with a plain ``urllib`` GET.

Why this exists
---------------
The Booster Robotics SDK exposes locomotion and odometry over DDS but does
NOT bind any camera topic in Python (verified: ``dir(booster_robotics_sdk_python)``
has no Vision/Image symbols). The camera stream is published only via the
robot's own ROS2 stack at ``/opt/booster/BoosterRos2Interface``. So the
laptop can't subscribe directly — but it can hit a tiny HTTP shim.

Endpoints
---------
- ``GET /frame.jpg``  Single JPEG of the latest frame. Returns 503 with a
                      retry-after header if no frame has arrived yet.
                      The laptop's ``MJPEGImageSource`` does a single GET
                      per call, so this is the endpoint to point it at.

- ``GET /stream``     ``multipart/x-mixed-replace`` MJPEG stream. Useful
                      to point a browser at for live diagnostics. NOTE:
                      the laptop's ``MJPEGImageSource`` calls ``read()``
                      which blocks until EOF, so do NOT point the laptop
                      at this endpoint — it would time out after 2 s.

- ``GET /``           Plain-text status: topic, message type, frame count,
                      last-frame age, resolution.

Running
-------
On the robot::

    ssh booster@192.168.10.102
    source /opt/booster/BoosterRos2Interface/install/setup.bash
    python3 robot_video_bridge.py --topic /booster_video_stream --port 8080

Then on the laptop::

    python3 navila_laptop_relay.py --mode print \\
        --image-source mjpeg \\
        --mjpeg-url http://192.168.10.102:8080/frame.jpg \\
        --server <desktop> --instruction "walk to the chair"

Auto-detects the topic message type (``sensor_msgs/msg/CompressedImage``
or ``sensor_msgs/msg/Image``) — first via ``get_topic_names_and_types``,
then falls back to whichever is requested on the CLI.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required. `pip install numpy` (or `apt install python3-numpy`).")

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required for JPEG encode/decode. "
             "`pip install opencv-python` (or `apt install python3-opencv`).")

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy,
    )
except ImportError:
    sys.exit("rclpy not importable. Did you `source /opt/booster/"
             "BoosterRos2Interface/install/setup.bash` first?")


# ---------------------------------------------------------------------- state


class FrameState:
    """Holds the latest JPEG-encoded frame plus diagnostic counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._jpeg: Optional[bytes] = None
        self._wh: Optional[tuple[int, int]] = None
        self._count = 0
        self._last_at: float = 0.0

    def update(self, jpeg: bytes, wh: tuple[int, int]) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._wh = wh
            self._count += 1
            self._last_at = time.monotonic()
            self._cond.notify_all()

    def latest(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def latest_with_age(self) -> tuple[Optional[bytes], float]:
        """Return (latest_jpeg, age_in_seconds). Age is monotonic-clock
        seconds since the callback last updated the slot — i.e. how long
        the bridge has been sitting on this frame. -1 if no frame yet."""
        with self._lock:
            if self._jpeg is None or self._last_at == 0.0:
                return None, -1.0
            return self._jpeg, time.monotonic() - self._last_at

    def wait_for_new(self, since_count: int, timeout_s: float = 2.0
                      ) -> tuple[Optional[bytes], int]:
        """Block until the frame counter advances past ``since_count``."""
        with self._cond:
            self._cond.wait_for(lambda: self._count > since_count,
                                 timeout=timeout_s)
            return self._jpeg, self._count

    def status(self) -> dict:
        with self._lock:
            return {
                "count": self._count,
                "wh": self._wh,
                "age_s": (time.monotonic() - self._last_at
                          if self._last_at else None),
            }


# ---------------------------------------------------------------- ROS2 → JPEG


def _image_to_bgr(msg) -> np.ndarray:
    """Convert sensor_msgs/Image to a BGR uint8 ndarray without cv_bridge.

    Supports the common encodings emitted by ROS2 camera nodes. Anything
    exotic (bayer, 16-bit, yuv) raises with a clear error so the user
    knows to add a case.
    """
    enc = msg.encoding.lower()
    h, w = msg.height, msg.width
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("bgr8", "8uc3"):
        return data.reshape(h, w, 3)
    if enc == "rgb8":
        return data.reshape(h, w, 3)[..., ::-1].copy()
    if enc == "bgra8":
        return cv2.cvtColor(data.reshape(h, w, 4), cv2.COLOR_BGRA2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(data.reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
    if enc in ("mono8", "8uc1"):
        return cv2.cvtColor(data.reshape(h, w), cv2.COLOR_GRAY2BGR)
    if enc in ("yuv422", "yuv422_yuy2", "yuyv"):
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    if enc == "nv12":
        # YUV 4:2:0 semi-planar: H*W Y plane + (H/2)*W interleaved UV plane.
        # Total bytes = 3/2 * H * W.
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
    if enc == "nv21":
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV21)

    raise RuntimeError(
        f"Unsupported Image encoding {msg.encoding!r}. Add a case in "
        "_image_to_bgr() or use cv_bridge.")


def make_image_cb(state: FrameState, quality: int):
    def cb(msg):
        try:
            bgr = _image_to_bgr(msg)
            ok, buf = cv2.imencode(".jpg", bgr,
                                    [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                return
            state.update(buf.tobytes(), (bgr.shape[1], bgr.shape[0]))
        except Exception as e:
            print(f"[bridge] Image cb failed: {e!r}", file=sys.stderr)
    return cb


def make_compressed_cb(state: FrameState, quality: int):
    def cb(msg):
        try:
            fmt = (msg.format or "").lower()
            data = bytes(msg.data)
            if "jpeg" in fmt or "jpg" in fmt:
                # Pass-through. We still want the dimensions for /status —
                # decode the header lazily by reading SOFn marker.
                wh = _peek_jpeg_dimensions(data)
                state.update(data, wh)
                return
            # Try generic decode for png/bmp/etc and re-encode.
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(
                    f"Unsupported CompressedImage format {msg.format!r}")
            ok, buf = cv2.imencode(".jpg", bgr,
                                    [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                return
            state.update(buf.tobytes(), (bgr.shape[1], bgr.shape[0]))
        except Exception as e:
            print(f"[bridge] CompressedImage cb failed: {e!r}", file=sys.stderr)
    return cb


def _peek_jpeg_dimensions(data: bytes) -> Optional[tuple[int, int]]:
    """Read W/H from JPEG SOFn marker without decoding the whole image."""
    i = 0
    n = len(data)
    if n < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    i = 2
    while i + 8 < n:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        # SOF0..SOF15 (skip SOF4=DHT and SOF8=JPG-internal: not SOFs)
        if marker in (0xC0, 0xC1, 0xC2, 0xC3,
                      0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB,
                      0xCD, 0xCE, 0xCF):
            h = (data[i + 5] << 8) | data[i + 6]
            w = (data[i + 7] << 8) | data[i + 8]
            return (w, h)
        if marker in (0xD8, 0xD9):
            return None
        length = (data[i + 2] << 8) | data[i + 3]
        i += 2 + length
    return None


# ---------------------------------------------------------------- topic probe


def detect_topic_type(node: Node, topic: str, timeout_s: float
                       ) -> Optional[str]:
    """Return the fully-qualified type, or None if not discovered in time."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for name, types in node.get_topic_names_and_types():
            if name == topic:
                for t in types:
                    if "CompressedImage" in t or "Image" in t:
                        return t
                return types[0] if types else None
        time.sleep(0.2)
    return None


# ---------------------------------------------------------------- HTTP server


def make_handler(state: FrameState, topic_name: str, type_name: str):
    BOUNDARY = "frameboundary"

    class Handler(BaseHTTPRequestHandler):
        # quieter access log; comment out to see every request
        def log_message(self, fmt, *args):
            sys.stderr.write(f"[http] {self.address_string()} - "
                              f"{fmt % args}\n")

        def do_GET(self):
            if self.path.startswith("/frame.jpg"):
                self._serve_frame()
            elif self.path.startswith("/stream"):
                self._serve_stream()
            elif self.path == "/" or self.path.startswith("/status"):
                self._serve_status()
            else:
                self.send_error(404, "Try /frame.jpg, /stream, or /")

        def _serve_frame(self):
            # ALWAYS returns the most-recent frame held in FrameState.
            # FrameState is a single-slot atomic store (no queue), so by
            # construction this can never serve an older frame than what
            # the latest callback has produced.
            jpeg, age_s = state.latest_with_age()
            if jpeg is None:
                self.send_response(503)
                self.send_header("Retry-After", "1")
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"no frame yet\n")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            # End-to-end latency aid: how long this frame has been sitting
            # in the bridge before we wrote it to the wire. Anything > a
            # few hundred ms means the subscriber callback is being starved
            # (encoding too slow, GIL contention, or ROS queue buildup).
            self.send_header("X-Frame-Age-Ms", f"{age_s * 1000:.1f}")
            self.end_headers()
            try:
                self.wfile.write(jpeg)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _serve_stream(self):
            self.send_response(200)
            self.send_header(
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last_count = -1
            try:
                while True:
                    jpeg, last_count = state.wait_for_new(last_count,
                                                            timeout_s=5.0)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--" + BOUNDARY.encode() + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _serve_status(self):
            s = state.status()
            wh = s["wh"]
            age = s["age_s"]
            size_str = f"{wh[0]}x{wh[1]}" if wh else "?"
            age_str = f"{age:.3f}" if age is not None else "-"
            body = (
                f"booster_video_bridge\n"
                f"topic:     {topic_name}\n"
                f"type:      {type_name}\n"
                f"frames:    {s['count']}\n"
                f"size:      {size_str}\n"
                f"age_s:     {age_str}\n"
            )
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


# --------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--topic", default="/booster_video_stream",
                     help="ROS2 image topic to subscribe to.")
    ap.add_argument("--port", type=int, default=8080,
                     help="HTTP port to serve frames on.")
    ap.add_argument("--bind", default="0.0.0.0",
                     help="HTTP bind address.")
    ap.add_argument("--quality", type=int, default=80,
                     help="JPEG quality (1-100) when re-encoding raw images "
                          "or non-JPEG CompressedImage formats.")
    ap.add_argument("--detect-timeout", type=float, default=5.0,
                     help="Seconds to wait for the topic to appear in DDS "
                          "discovery before falling back to --force-type.")
    ap.add_argument("--force-type", choices=["auto", "compressed", "image"],
                     default="auto",
                     help="Override message-type auto-detection. "
                          "'compressed' = sensor_msgs/CompressedImage; "
                          "'image' = sensor_msgs/Image.")
    ap.add_argument("--qos-depth", type=int, default=1,
                     help="Subscriber queue depth. Default 1 = always-latest "
                          "(old frames are dropped before the callback ever "
                          "sees them). Raise only if you see frame loss in "
                          "/status counts and don't care about latency.")
    ap.add_argument("--qos-reliability", choices=["best_effort", "reliable"],
                     default="best_effort",
                     help="best_effort lets DDS drop frames under back-"
                          "pressure (correct for live camera). reliable will "
                          "retransmit, which can cause exactly the 10–15s "
                          "delay we are trying to avoid.")
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("booster_video_bridge")

    # ---- pick subscriber type
    type_name: Optional[str] = None
    if args.force_type == "auto":
        print(f"[bridge] probing {args.topic} for up to "
              f"{args.detect_timeout:.1f}s ...", flush=True)
        type_name = detect_topic_type(node, args.topic, args.detect_timeout)
        if type_name is None:
            print(f"[bridge] WARNING: {args.topic} not discovered in time. "
                  "Defaulting to CompressedImage. Pass --force-type image "
                  "if your camera publishes raw frames.", file=sys.stderr)
            type_name = "sensor_msgs/msg/CompressedImage"
    elif args.force_type == "compressed":
        type_name = "sensor_msgs/msg/CompressedImage"
    else:
        type_name = "sensor_msgs/msg/Image"

    # Always-latest QoS: depth=1 + BEST_EFFORT so DDS drops stale frames
    # at the wire instead of queueing them; combined with FrameState's
    # single-slot store this gives us a true "newest frame wins" pipeline.
    rel = (ReliabilityPolicy.BEST_EFFORT
           if args.qos_reliability == "best_effort"
           else ReliabilityPolicy.RELIABLE)
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=args.qos_depth,
        reliability=rel,
        durability=DurabilityPolicy.VOLATILE,
    )

    state = FrameState()
    if "CompressedImage" in type_name:
        from sensor_msgs.msg import CompressedImage
        node.create_subscription(CompressedImage, args.topic,
                                  make_compressed_cb(state, args.quality),
                                  qos)
    else:
        from sensor_msgs.msg import Image
        node.create_subscription(Image, args.topic,
                                  make_image_cb(state, args.quality),
                                  qos)

    print(f"[bridge] subscribed to {args.topic} as {type_name} "
          f"(qos: depth={args.qos_depth}, rel={args.qos_reliability})",
          flush=True)

    # ---- start HTTP server in a daemon thread
    handler = make_handler(state, args.topic, type_name)
    httpd = ThreadingHTTPServer((args.bind, args.port), handler)
    http_thread = threading.Thread(target=httpd.serve_forever,
                                    name="http-server", daemon=True)
    http_thread.start()
    print(f"[bridge] HTTP listening on http://{args.bind}:{args.port}/  "
          "(try /frame.jpg, /stream, or /)", flush=True)

    # ---- spin ROS2 on the main thread until Ctrl-C
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[bridge] Ctrl-C — shutting down.")
    finally:
        httpd.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
