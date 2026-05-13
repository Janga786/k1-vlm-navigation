"""TCP framing for the desktop NaVILA server ↔ laptop relay link.

Wire format for one message:

    [4 bytes BE uint32: json_len]
    [json_len bytes: utf-8 JSON header]
    [4 bytes BE uint32: blob_len]
    [blob_len bytes: binary blob, may be empty]

The JSON header always carries a ``"type"`` field. The blob carries
JPEG bytes for ``"tick"`` messages.

Message types (all JSON header fields):

LAPTOP → SERVER
---------------
- ``hello``  client says hi:
    ``{"type": "hello", "client": "laptop-relay", "version": 1}``

- ``set_instruction``  start / restart a multi-step plan:
    ``{"type": "set_instruction", "instruction": "walk forward | turn left 90 deg",
       "per_step_time": 25.0, "action_duration": 1.5,
       "vx_max": 0.4, "vy_max": 0.15, "vyaw_max": 0.4,
       "heading_assist": false, "turn_controller": false,
       "proximity_threshold": 1.0}``

- ``tick``  push a frame and pull the current command:
    ``{"type": "tick", "have_image": true, "have_pose": false,
       "pose_xy": null, "pose_yaw": null}``
    blob = JPEG bytes if ``have_image`` else empty.

- ``shutdown``  client is going away cleanly:
    ``{"type": "shutdown"}``

SERVER → LAPTOP
---------------
- ``hello_ack``:
    ``{"type": "hello_ack", "server": "navila-server", "model_loaded": true,
       "version": 1}``

- ``instruction_ack``:
    ``{"type": "instruction_ack", "ok": true, "step_count": 3,
       "steps": [{"instruction": "...", "yaw_target_deg": null,
                  "proximity_target": null, "time_limit": 25.0}, ...]}``

- ``state``:
    ``{"type": "state",
       "vx": 0.4, "vy": 0.0, "vyaw": 0.0,
       "tag": "VLM ",                 # VLM | HEAD | TURN | DRAIN
       "label": "forward 0.75m",
       "raw": "move forward 75 cm",
       "step_idx": 0, "step_total": 3,
       "step_instruction": "walk forward",
       "step_advanced": false,
       "done_reason": null,
       "all_done": false,
       "vlm_stop": false,
       "inf_count": 17, "inf_ms": 412.3,
       "buffer_size": 8}``

- ``error``:
    ``{"type": "error", "message": "..."}``
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Optional


DEFAULT_PORT = 5555

# Hard upper bounds — protect both ends from a corrupted length header.
MAX_JSON_BYTES = 1 * 1024 * 1024        # 1 MiB header
MAX_BLOB_BYTES = 16 * 1024 * 1024       # 16 MiB blob (way more than a JPEG)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"socket closed while reading "
                f"({len(buf)}/{n} bytes received)"
            )
        buf.extend(chunk)
    return bytes(buf)


def send_msg(sock: socket.socket, header: dict, blob: bytes = b"") -> None:
    """Send ``header`` (JSON) and optional binary ``blob``.

    Single ``sendall`` per piece — TCP handles fragmentation. The two
    length prefixes let the receiver allocate buffers up front.
    """
    json_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(json_bytes) > MAX_JSON_BYTES:
        raise ValueError(f"JSON header too large: {len(json_bytes)} bytes")
    if len(blob) > MAX_BLOB_BYTES:
        raise ValueError(f"blob too large: {len(blob)} bytes")
    sock.sendall(struct.pack(">I", len(json_bytes)))
    sock.sendall(json_bytes)
    sock.sendall(struct.pack(">I", len(blob)))
    if blob:
        sock.sendall(blob)


def recv_msg(sock: socket.socket) -> tuple[dict, bytes]:
    """Read one (header, blob) pair from ``sock``.

    Raises ``ConnectionError`` if the peer closes mid-message, and
    ``ValueError`` if either length exceeds the configured caps.
    """
    json_len = struct.unpack(">I", _recv_exact(sock, 4))[0]
    if json_len > MAX_JSON_BYTES:
        raise ValueError(f"announced JSON size {json_len} exceeds cap")
    json_bytes = _recv_exact(sock, json_len)
    header = json.loads(json_bytes.decode("utf-8"))

    blob_len = struct.unpack(">I", _recv_exact(sock, 4))[0]
    if blob_len > MAX_BLOB_BYTES:
        raise ValueError(f"announced blob size {blob_len} exceeds cap")
    blob = _recv_exact(sock, blob_len) if blob_len > 0 else b""
    return header, blob


def connect(host: str, port: int = DEFAULT_PORT,
            connect_timeout: float = 10.0,
            io_timeout: Optional[float] = None) -> socket.socket:
    """Open a TCP socket to (host, port), with the given timeouts.

    ``io_timeout=None`` keeps the socket blocking after connect; the
    client typically wants a finite read timeout so a hung server
    doesn't lock up the relay loop.
    """
    s = socket.create_connection((host, port), timeout=connect_timeout)
    s.settimeout(io_timeout)
    # TCP_NODELAY: we send small JSON headers; don't wait for Nagle.
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s


def listen(host: str, port: int = DEFAULT_PORT,
           backlog: int = 1) -> socket.socket:
    """Open a TCP listening socket. Single-laptop deployment: backlog=1."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(backlog)
    return s
