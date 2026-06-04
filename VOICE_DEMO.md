# Voice-controlled NaVILA demo — 3-terminal launch sequence

Voice is the optional input front-end to the existing NaVILA pipeline. The
desktop NaVILA server is unchanged; only the laptop relay swaps in
`navila_voice_relay.py`.

```
laptop mic   →  Whisper (local)  →  Groq llama-3.1-8b-instant  →  set_instruction
                                                                       │
                                                              desktop NaVILA server
                                                                       │
              tick loop  ←  vx/vy/vyaw  ←  NaVILA VLM (on RTX 5090)
                  │
            B1LocoClient.Move()
                  │
                K1 walks
                  │
            espeak-ng "Task complete." → next voice command
```

---

## One-time laptop setup

```bash
sudo apt install -y portaudio19-dev ffmpeg sshpass espeak-ng
# Match the desktop env's pip versions (use the laptop's relay venv):
source ~/Projects/k1_research/.venv-relay/bin/activate
pip install sounddevice scipy openai-whisper groq

# Persist the Groq key (already done on the desktop):
echo 'export GROQ_API_KEY="<your-key>"' >> ~/.bashrc
source ~/.bashrc

# Smoke-test the mic + Whisper path BEFORE involving the robot:
cd ~/Projects/k1_research/experiments/navila
python test_voice_capture.py --list-mics
python test_voice_capture.py            # speak a sentence, see transcript
```

---

## Demo (3 terminals)

### Terminal 1 — Desktop: NaVILA server

```bash
# Already running from Task 1, but to restart cleanly:
pkill -f navila_server.py 2>/dev/null
nohup bash ~/start_navila_server.sh > /tmp/navila_server.log 2>&1 &
disown
tail -F /tmp/navila_server.log         # wait for "listening on 0.0.0.0:5555"
```

Default port `5555`. Tailscale-reachable from the laptop.

### Terminal 2 — Robot: camera bridge

The voice relay uses the existing camera path (MJPEG over Ethernet from the
K1, or ZED on the laptop). For the MJPEG path the bridge runs on the K1's
onboard PC:

```bash
ssh booster@192.168.10.102
# On the robot:
source /opt/ros/humble/setup.bash
python3 ~/robot_video_bridge.py \
    --topic /booster_video_stream --port 8080
# Leaves an MJPEG stream at http://192.168.10.102:8080/stream
```

(If you have the ZED plugged into the laptop, skip this terminal and pass
`--image-source zed` on the laptop side instead.)

### Terminal 3 — Laptop: voice relay

```bash
cd ~/Projects/k1_research/experiments/navila

# Print-only smoke test (no robot, no SDK — just verifies the loop)
python navila_voice_relay.py \
    --server <desktop-tailscale-ip> \
    --mode print --image-source static

# Robot dry-run (B1LocoClient connects, Move() is logged not sent)
python navila_voice_relay.py \
    --server <desktop-tailscale-ip> \
    --mode dry --net 192.168.10.102 \
    --image-source mjpeg --mjpeg-url http://192.168.10.102:8080/stream

# Live deploy (K1 WALKS)
python navila_voice_relay.py \
    --server <desktop-tailscale-ip> \
    --mode live --net 192.168.10.102 \
    --image-source mjpeg --mjpeg-url http://192.168.10.102:8080/stream
```

### What you'll see in Terminal 3

```
[voice] loading Whisper (base)...
[voice] Whisper ready.
[voice] >>> Voice control ready. Press enter to speak.

[voice] press ENTER to start recording (Ctrl-C to exit voice mode)...
<ENTER>
[voice] RECORDING — press ENTER to stop.
<speak: "walk to the volleyball and stop">
<ENTER>
[voice] captured 2.1s (33600 samples).
[voice] transcript: 'Walk to the volleyball and stop.'
[voice] >>> I heard: Walk to the volleyball and stop. Executing.
[voice] → NaVILA set_instruction: 'walk to the volleyball | stop'
[remote] connecting to <ip>:5555 (timeout=5.0s) ...
[remote] handshake ok — server='navila-server' model_loaded=True
[remote] instruction set — 2 sub-step(s):
  step 1/2: 'walk to the volleyball'  tlim=25s
  step 2/2: 'stop'  tlim=25s
[voice t=  1.0s] step 1/2 [VLM ] vx=+0.40 vy=+0.00 vyaw=-0.10 raw='walk forward'
...
[remote] sub-step advanced → 2/2  reason='done'
...
[voice] result: complete
[voice] >>> Task complete.

[voice] press ENTER to start recording...
```

The cv2 HUD window opens automatically (suppressed with `--no-display`).

---

## Voice loop control

| Action | How |
|---|---|
| Start recording | `ENTER` |
| Stop recording | `ENTER` again |
| Emergency-stop the K1 mid-instruction | `Ctrl-C` (zeroes Move, loops back to listening) |
| Exit voice mode entirely | `Ctrl-C` at the recording prompt, or `Ctrl-D`/EOF |
| Quiet TTS | `--no-tts` (speech still prints to stdout) |
| Skip Groq cleanup | `--no-groq-cleanup` (raw Whisper transcript → NaVILA) |

Saying the single word **"stop"** (or "please stop", "robot stop") bypasses
Groq for a faster stop.

---

## Falling back to the old one-shot text mode

```bash
python navila_voice_relay.py \
    --input-mode text \
    --instruction "walk to the chair | turn right 90 deg" \
    --server <desktop-tailscale-ip> --mode print --image-source static
```

This sends `--instruction` once and exits when `all_done` (matches
`navila_laptop_relay.py`'s behavior).

`navila_laptop_relay.py` itself is **untouched** — it still works exactly as
before.

---

## Vocabulary the Groq cleanup will produce

The Groq system prompt restricts output to NaVILA-compatible primitives:

| What you say | Groq → COMMAND: ... |
|---|---|
| "walk to the volleyball and stop" | `walk to the volleyball \| stop` |
| "turn around"                     | `turn left 180 deg` |
| "back up a little"                | `walk backward` |
| "go to the chair then turn right" | `walk to the chair \| turn right 90 deg` |
| "stop"                            | `stop` (handled before Groq) |
| "what do you see?"                | (no command — Groq returns a REPLY: ...) |

Movement vocabulary is the same one the desktop server parses in
`navila_k1_core.py`; see `experiments/navila/README.md` for the full list.
