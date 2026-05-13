# NaVILA Remote Deployment — Desktop-server / Laptop-relay

Two-process split for running NaVILA on the K1 when your GPU lives at home
and your robot lives at the lab.

```
   home                                     lab
 ┌─────────────────────────┐          ┌────────────────────────────┐
 │      DESKTOP            │          │         LAPTOP             │
 │  RTX 5090 + NaVILA      │          │  no GPU, ZED + cv2 HUD     │
 │                         │          │                            │
 │  navila_server.py       │  TCP     │  navila_laptop_relay.py    │
 │  port 5555  ◀──────────────────────────  socket client          │
 │  ▲                      │ Tailscale│                            │
 │  │ frames+pose          │          │  ▲    ▼                    │
 │  │ vx,vy,vyaw + text    │          │  │    │  B1LocoClient.Move │
 │  ▼                      │          │  │    ▼                    │
 │  NaVILA + 8-frame buf   │          │  ZED   ┌─────────────┐     │
 │  multi-step planner     │          │  cam   │   K1 ROBOT  │     │
 │  controllers            │          │  ────► │ 192.168.10.102    │
 └─────────────────────────┘          └────────────────────────────┘
```

**No torch, no llava, no NaVILA weights on the laptop.** Only the
Booster SDK, OpenCV, and Pillow.

## Files

| file | runs on | what it does |
|---|---|---|
| `navila_server.py` | desktop | loads NaVILA once, TCP-listens on 5555, runs inference + multi-step planner per session |
| `navila_laptop_relay.py` | laptop | grabs head-cam frames, ships JPEGs to the server, drives the K1 via `B1LocoClient.Move`, displays the HUD |
| `navila_protocol.py` | both | length-prefixed JSON-+-blob framing |
| `setup_laptop.sh` | laptop | installs minimal deps + Booster SDK |

## 1. Bootstrap the laptop (once)

```bash
# On the laptop:
mkdir -p ~/Projects/k1_research/experiments/navila
# Either rsync directly from the desktop ...
rsync -av janga@desktop.tail-XXX.ts.net:~/Projects/k1_research/experiments/navila/ \
    ~/Projects/k1_research/experiments/navila/
# ... or use the bundled script with --rsync-from:
cd ~/Projects/k1_research/experiments/navila
./setup_laptop.sh --rsync-from janga@desktop.tail-XXX.ts.net
```

The setup script:
- creates `$K1RES_DIR/.venv-relay` (pass `--no-venv` to skip)
- `apt install`s the SDK build deps (cmake, asio, tinyxml2, etc.)
- `pip install`s `numpy`, `pillow`, `opencv-python`, `booster_robotics_sdk_python`
- falls back to building the SDK from source if the pip wheel isn't available
- verifies the imports

If you don't want it touching apt or pip, pass `--skip-sdk` and/or
`--skip-pip` and install by hand.

## 2. Start the desktop server

```bash
# On the desktop (with the navila conda env or wherever NaVILA can import):
cd ~/Projects/k1_research/experiments/navila
python navila_server.py --bind 0.0.0.0 --port 5555
```

It loads the model from
`~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f`
(override with `--model-path`).

Expected output:

```
[server] loading NaVILA from /home/janga/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f (this is slow)...
[VLM] loading NaVILA from .../navila-llama3-8b-8f ...
[VLM] NaVILA ready.
[server] model loaded and inference thread running.
[server] listening on 0.0.0.0:5555 — Ctrl-C to exit.
```

If you only want Tailscale-side access, replace `0.0.0.0` with your
desktop's Tailscale IP and let your firewall reject the rest.

## 3. Start the laptop relay

The relay mirrors the three-mode progression of `navila_k1_realrobot.py`:
**print → dry → live**.  Walk up the ladder.

```bash
# On the laptop (venv active if you used one):
source ~/Projects/k1_research/.venv-relay/bin/activate
cd ~/Projects/k1_research/experiments/navila

DESKTOP_IP=100.x.y.z   # your desktop's Tailscale IP

# (a) PRINT mode — no SDK at all. End-to-end network + planner smoke test.
python navila_laptop_relay.py --mode print \
    --image-source static \
    --image-path ~/Projects/k1_research/experiments/vla/test_image.jpg \
    --server "$DESKTOP_IP" \
    --instruction "walk forward 3 meters"

# (b) DRY mode — SDK init + connect to K1, Move() is logged not sent.
python navila_laptop_relay.py --mode dry \
    --image-source mjpeg \
    --mjpeg-url http://192.168.10.102:8080/stream \
    --server "$DESKTOP_IP" \
    --net 192.168.10.102 \
    --instruction "walk forward 3 meters | turn left 90 deg"

# (c) LIVE mode — floor cleared, K1 IS WALKING. Emergency-stop: 'q' or Ctrl-C.
python navila_laptop_relay.py --mode live \
    --image-source mjpeg \
    --mjpeg-url http://192.168.10.102:8080/stream \
    --server "$DESKTOP_IP" \
    --net 192.168.10.102 \
    --instruction "walk to the chair | turn right 90 deg | walk forward"
```

While the HUD is up:
- press **`q`** for an emergency stop (zeros `Move`, switches K1 to `kDamping`,
  closes the socket, exits)
- **Ctrl-C** does the same via SIGINT
- if the desktop crashes / Tailscale drops, the laptop's tick thread errors
  out within `--io-timeout` seconds AND the `LiveActuator` watchdog
  zeroes the cmd within 1.5 s — either way the robot stops

## HUD layout

```
┌──────────────────────────────────────────────────────────────┐
│ [LIVE] walk to the chair | turn right 90 deg | walk forward  │
│ step 1/3: walk to the chair                                  │
│ NaVILA: 'move forward 75 cm'                                 │
│                                                              │
│                                                              │
│                  (live camera feed)                          │
│                                                              │
│                                                              │
│                                                              │
│ out: vx=+0.40 vy=+0.00 vyaw=+0.00 [VLM ]                     │
│ vlm: inf#17 412ms  buf=8  stop=False  link_age=0.31s  grab=8ms │
│ press 'q' to emergency-stop                                  │
└──────────────────────────────────────────────────────────────┘
```

- `link_age` is "seconds since the last server response."  Watch this —
  if it climbs above 1 s, your link is slow or NaVILA is wedged.
- `inf#` is the cumulative NaVILA inference count; it should tick up about
  once per second.

## Image-source decision tree

The laptop doesn't have the ZED plugged in — it's on the K1.  Two options:

1. **MJPEG stream from the K1 onboard PC.**  Run a tiny ZED→MJPEG bridge
   on the K1's onboard computer (the K1 SDK examples don't include one
   out of the box; the Stereolabs SDK's ZED_SVO_Recorder or a 50-line
   gst-launch pipeline both work).  Then on the laptop pass
   `--image-source mjpeg --mjpeg-url http://192.168.10.102:8080/stream`.
2. **Plug the ZED into the laptop directly** (e.g. for bench tests
   without the robot).  Install pyzed on the laptop and pass
   `--image-source zed`.  Pyzed is not installed by `setup_laptop.sh` —
   grab it from Stereolabs's release page when you need it.

For end-to-end smoke testing without any camera, use
`--image-source static --image-path some.jpg` or
`--image-source dir --image-dir recorded_frames/`.

## Latency budget

| stage | budget | notes |
|---|---|---|
| ZED grab | ~10 ms | VGA / 30 fps |
| JPEG encode (laptop) | ~5 ms | quality 80, ~30 KB at VGA |
| Tailscale RTT | 20–80 ms | depends on routing; direct = lower |
| NaVILA inference | 350–500 ms | on a 5090, batched 8 frames |
| planner + parse | <1 ms | trivial |
| K1 `Move()` send | 1–2 ms | over Ethernet |

The relay's `--tick-period` defaults to 0.4 s — that's how often a fresh
frame is pushed to the server.  Anything between 0.3 and 0.6 s is fine;
go lower only if your link is fast.

NaVILA already throttles itself at ~1 Hz inference; sending more frames
than that just refreshes the rolling buffer, which is harmless.

## Pre-flight pyramid (before going live)

Climb each rung — don't skip:

1. **Unit tests on the desktop:**
   `cd experiments/navila && python -m unittest discover tests -v`
   (58 tests, ~5 s.)
2. **Server boots:**  start `navila_server.py`; confirm the "NaVILA ready"
   line shows up.  This catches checkpoint / CUDA / torch issues.
3. **Print-mode smoke test (laptop, no robot):**  `--mode print
   --image-source static`.  Confirm sub-step transitions in the HUD;
   you should see `step 1/N: ...` etc.
4. **Dry-mode on the powered K1 (laptop, robot in a safe stance):**
   `--mode dry --image-source mjpeg`.  Confirm the K1 SDK init succeeds
   ("[actuator/dry] B1LocoClient initialised") and `Move()` calls are
   logged but never sent.
5. **Live deploy, floor cleared, hand on kill switch:**  `--mode live`.
   Test instruction: `"walk forward 1 meter"`.  Verify it stops within
   1.5 s of `q` press.
6. **Then** try the multi-step instructions.

## Troubleshooting

| symptom | likely cause | fix |
|---|---|---|
| `[remote] connecting to ...` hangs | desktop firewall, wrong IP, or server not running | check `ss -tlnp` on desktop, `tailscale ping <desktop>` on laptop |
| `connection refused` | server not started | bring `navila_server.py` up first |
| `[remote] network error: timed out` | NaVILA hung, Tailscale exit-node issue | restart server; check `--io-timeout`; `tailscale status` |
| HUD says "buf=0" forever | the laptop is sending frames but the server isn't decoding | check JPEG quality / corrupted JPEGs; pass `--jpeg-quality 90` |
| robot drifts left/right on "walk forward" | known NaVILA bias; no heading assist without pose | acceptable for short distances; needed a `PoseSource` for assist |
| robot spins on a "turn left N deg" | `--turn-controller` is on but no pose source | turn it off (`--no-turn-controller`, the default) OR plumb pose |
| `import booster_robotics_sdk_python: No module named ...` | SDK wheel not available for your arch | run `./setup_laptop.sh` again with the source-build fallback, or `--mode print` |
| HUD window doesn't appear | running headless / X forward not set | pass `--no-display` to fall back to console |

## What stays on the desktop

- NaVILA weights (`~/Projects/k1_research/booster/NaVILA/checkpoints/`)
- The conda env with torch / llava / mujoco
- All sim work (`navila_k1_walking_loop.py`)
- This server (`navila_server.py`)

## What needs to live on the laptop

- `navila_protocol.py`, `navila_laptop_relay.py`
- `navila_k1_realrobot.py` (imported by the relay for `ImageSource` / `Actuator` classes)
- `navila_k1_core.py`, `navila_k1_bridge.py` (transitive imports — neither
  triggers torch at module load)
- Booster SDK + Python bindings
- numpy + Pillow + opencv-python

Total laptop footprint: ~30 MB Python deps + the SDK.  No GPU, no model.
