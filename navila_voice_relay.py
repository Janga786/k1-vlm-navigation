#!/usr/bin/env python3
"""Voice-controlled NaVILA laptop relay.

Same protocol as ``navila_laptop_relay.py`` (the desktop NaVILA server is
unchanged), but the ``--instruction`` is captured from the laptop microphone
each iteration instead of being hard-coded at startup.

Pipeline (per voice turn)::

    1. Push-to-talk: press ENTER, speak, press ENTER again to stop.
    2. Whisper transcribes the WAV locally (no network, no GPU req).
    3. Groq (llama-3.1-8b-instant) cleans the transcript into a NaVILA
       command (or returns a conversational reply for non-movement input).
       Use ``--no-groq-cleanup`` to send the raw transcript instead.
    4. espeak-ng on the laptop speaks ``"I heard <X>. Executing."``.
    5. set_instruction is sent to the desktop NaVILA server; the standard
       tick loop runs until ``all_done``.
    6. espeak-ng says ``"Task complete."`` and we loop back to step 1.

The actuator + image-source + planner flags are identical to
``navila_laptop_relay.py``. A ``--input-mode text`` path is kept for
non-voice testing: behaves exactly like the old relay (one ``--instruction``
sent at start, run to completion, exit).

Examples
========

End-to-end voice loop using a static test frame (no robot)::

    python navila_voice_relay.py --mode print \\
        --image-source static \\
        --server <desktop-tailscale-ip>

Live deploy on the K1 with ZED camera + voice::

    python navila_voice_relay.py --mode live \\
        --image-source zed --net 192.168.10.102 \\
        --server <desktop-tailscale-ip>

One-shot text mode (matches the old relay's behavior)::

    python navila_voice_relay.py --input-mode text \\
        --instruction "walk to the chair" \\
        --mode print --server <desktop-tailscale-ip>
"""

from __future__ import annotations

import argparse
import io
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import navila_protocol as proto  # noqa: E402
from navila_k1_bridge import ACTION_DURATION  # noqa: E402

# Reuse the existing relay's components — RemoteVLMClient handles all
# the wire protocol details, including multiple set_instruction calls
# on the same connection (the desktop server explicitly supports this).
from navila_laptop_relay import (  # noqa: E402
    RemoteVLMClient,
    encode_jpeg,
    make_image_source,
    make_actuator,
    _clip,
    draw_hud,
)


# ============================================================================
# Voice-side config + Groq system prompt
# ============================================================================

SAMPLE_RATE = 16000
# Default Whisper model. 'base' is 74M params, ~150 MB download, and
# transcribes a 2-3 s push-to-talk clip on CPU in well under a second.
# 'medium' (the previous default) was 769M params; on a laptop CPU it
# pinned every core for 10-30 s and FROZE the machine while it ran.
# Don't change this default without re-testing on a laptop. The Groq
# LLM cleanup absorbs minor transcription wobble, so 'base' is fine
# for short navigation commands ("walk to the chair", "turn left 90").
DEFAULT_WHISPER = "base"
WHISPER_CHOICES = ("tiny", "base", "small", "medium", "large")
# Whisper device. CPU is the right default for the laptop relay: the
# laptop typically has a small GPU shared with the desktop env (we OOM'd
# at ~52 MB free trying to load 'medium' on CUDA), and CPU is fast
# enough for the short push-to-talk clips this script captures.
DEFAULT_WHISPER_DEVICE = "cpu"
WHISPER_DEVICE_CHOICES = ("cpu", "cuda", "auto")
# Maximum CPU threads PyTorch can use for Whisper inference. Without a
# cap, PyTorch grabs all cores and the OS becomes unresponsive while a
# transcribe is in flight. 4 is a sweet spot for laptops: most of the
# speedup is in the first few threads, and we keep cores free for the
# rest of the system.
DEFAULT_CPU_THREADS = 4
DEFAULT_GROQ = "llama-3.1-8b-instant"

GROQ_SYS_PROMPT = """You are a translator. You convert a spoken instruction \
into ONE NaVILA navigation command, and output NOTHING ELSE.

OUTPUT FORMAT (strict):
  - One single line, ending immediately. No newlines. No second line.
  - No prose, no explanation, no greeting, no "REPLY:", no "COMMAND:",
    no quotes, no markdown, no trailing punctuation.
  - If the user's request is not a navigation instruction, output exactly:
        stop

NaVILA primitives (chain with " | "):
  walk forward
  walk backward
  turn left N deg            (e.g. "turn left 90 deg")
  turn right N deg
  walk to the <object>       (vision-grounded)
  stop

Examples (input -> output, OUTPUT IS THE ENTIRE LINE):
  "go to the volleyball and stop"      -> walk to the volleyball | stop
  "walk to the volleyball"             -> walk to the volleyball
  "turn around"                        -> turn left 180 deg
  "back up a little"                   -> walk backward
  "walk to the chair then turn right"  -> walk to the chair | turn right 90 deg
  "stop"                               -> stop
  "what do you see"                    -> stop
  "hello"                              -> stop

Remember: output ONLY the command line. Anything else breaks the robot."""


# Regexes for stripping leftover labels / filler from the Groq output.
# Groq sometimes prepends "COMMAND:" / "Command:" / "Output:" anyway, or
# tacks on a second "REPLY: ..." line. We aggressively strip both.
_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:command|reply|output|answer|response|result)\s*:\s*",
    re.IGNORECASE,
)
_REPLY_INLINE_RE = re.compile(
    r"\s*\b(?:reply|note|comment)\s*:.*$", re.IGNORECASE
)
# Filler phrases models like to add — "Sure!", "Okay,", "Here you go:", etc.
_FILLER_PREFIX_RE = re.compile(
    r"^\s*(?:sure|okay|ok|alright|got\s+it|here\s+(?:is|you\s+go)|"
    r"of\s+course|certainly)[\s,!.:-]+",
    re.IGNORECASE,
)
# Trailing quotes / markdown / punctuation that NaVILA's parser would
# otherwise see as part of the instruction.
_TRIM_CHARS = " \t\r\n.\"'`*_-"


def clean_groq_output(raw: str) -> str:
    """Sanitize the Groq response into a single NaVILA-compatible command.

    - Strips everything after the first newline (Groq frequently adds
      a second "REPLY: ..." line even when told not to).
    - Strips leading labels like "COMMAND:", "Reply:", "Output:".
    - Strips conversational filler ("Sure!", "Okay,", "Here you go:").
    - Strips an inline trailing "REPLY: ..." / "Note: ...".
    - Strips wrapping quotes / markdown / trailing punctuation.
    """
    if not raw:
        return ""
    # First line only — kills any second "REPLY:" line.
    s = raw.splitlines()[0]
    # Iteratively peel labels + filler (handles "Command: Sure! walk ...").
    for _ in range(4):
        before = s
        s = _LABEL_PREFIX_RE.sub("", s)
        s = _FILLER_PREFIX_RE.sub("", s)
        if s == before:
            break
    # Drop trailing "REPLY: ..." / "Note: ..." on the same line.
    s = _REPLY_INLINE_RE.sub("", s)
    # Strip wrapping quotes / markdown / trailing punctuation.
    s = s.strip(_TRIM_CHARS)
    return s


# ============================================================================
# Audio capture — push-to-talk via two ENTER keypresses
# ============================================================================


def list_input_devices() -> str:
    """Returns a multi-line string describing input-capable devices."""
    import sounddevice as sd
    out = ["[voice] available input devices:"]
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            default = " (default)" if idx == sd.default.device[0] else ""
            out.append(f"  [{idx}] {dev['name']} "
                       f"({dev['max_input_channels']}ch, "
                       f"{int(dev['default_samplerate'])}Hz){default}")
    return "\n".join(out)


# Minimum recording window (seconds). Anything shorter is almost certainly
# the result of a queued/double-tapped ENTER — discard instead of trying
# to transcribe silence. The stop-ENTER loop also enforces this so a queued
# ENTER right after the start-ENTER can't end the recording at 0 seconds.
MIN_RECORD_S = 0.5
DISCARD_BELOW_S = 0.3
# Hard upper bound: recording ALWAYS stops at this many seconds, even if
# the stop-ENTER never arrives. This is a safety net against terminal /
# audio-backend weirdness that wedges sys.stdin.readline().
MAX_RECORD_S = 15.0
# How often to poll the stdin-watcher thread in the wait loop.
WAIT_POLL_S = 0.05
# Heartbeat interval — print "(recording... Xs)" this often so the user
# (and we, in bug reports) can see the loop is alive.
HEARTBEAT_S = 2.0
# Max time we'll wait for sounddevice to close the audio device before
# moving on. PortAudio occasionally hangs on close; we'd rather lose the
# device descriptor than freeze the demo.
STREAM_CLOSE_TIMEOUT_S = 3.0


def _flush_stdin() -> None:
    """Discard any bytes already buffered on stdin.

    Without this, leftover newlines from a previous turn (or from the
    user double-tapping ENTER) instantly satisfy the next readline(),
    which is why the old push-to-talk auto-started recording AND
    instantly stopped it. termios.tcflush is the only reliable way to
    drop bytes already sitting in the TTY driver's input queue.
    """
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        # stdin isn't a TTY (piped/redirected) or termios unavailable —
        # readline() will behave normally in that case.
        pass


def _readline_or_eof(prompt: Optional[str] = None) -> str:
    """sys.stdin.readline() that raises EOFError on a closed stdin.

    Plain readline() returns '' on EOF, which the old code didn't
    distinguish from a real ENTER — that's how a closed stdin produced
    an infinite "I didn't catch that" loop instead of a clean exit.
    """
    if prompt is not None:
        print(prompt, flush=True)
    line = sys.stdin.readline()
    if line == "":
        raise EOFError("stdin closed")
    return line


def _force_canonical_tty() -> Optional[list]:
    """Force the TTY back to canonical line mode with CR→NL translation.

    Audio backends (PortAudio / ALSA / PulseAudio under sounddevice) or
    the Booster SDK occasionally leave the terminal in a state where
    ENTER delivers only '\\r' or input isn't line-buffered. We re-enable
    ICRNL | ICANON | ECHO so the kernel actually delivers byte sequences
    on ENTER. Returns the previous termios attrs (pass to ``_restore_tty``)
    or None if stdin isn't a TTY.
    """
    try:
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[0] |= termios.ICRNL                    # iflag: CR -> NL
        new[3] |= termios.ICANON | termios.ECHO    # lflag
        termios.tcsetattr(fd, termios.TCSANOW, new)
        return old
    except Exception:
        return None


def _restore_tty(saved) -> None:
    if saved is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, saved)
    except Exception:
        pass


def _start_stdin_watcher(enter_event: "threading.Event",
                         eof_event: "threading.Event") -> None:
    """Spawn a daemon thread that reads raw bytes from stdin.

    Sets ``enter_event`` each time the read returns bytes containing
    '\\n' or '\\r'. Sets ``eof_event`` on EOF / read error. Uses
    ``os.read`` directly (NOT sys.stdin.readline) so terminal-mode
    quirks can't block forever waiting for a translated newline.
    """
    import os
    import threading

    fd = sys.stdin.fileno()

    def _watcher() -> None:
        try:
            while True:
                # Blocking read; returns whatever the TTY driver delivers
                # at the next line boundary (canonical mode) or on each
                # keystroke (raw mode). Either way, ENTER comes through.
                chunk = os.read(fd, 256)
                if not chunk:
                    eof_event.set()
                    return
                if b"\n" in chunk or b"\r" in chunk:
                    enter_event.set()
                    # Wait for the main thread to acknowledge this ENTER
                    # before reading more — otherwise rapid ENTERs all
                    # collapse into a single set().
                    while enter_event.is_set():
                        time.sleep(WAIT_POLL_S)
        except Exception:
            eof_event.set()

    threading.Thread(target=_watcher, name="stdin-watcher",
                     daemon=True).start()


def _wait_for_stop_enter(min_record_s: float,
                         max_record_s: float,
                         rec_start: float) -> str:
    """Wait for stop-ENTER, with min/max bounds and a hard timeout.

    Returns:
        "user-stop"    — user pressed ENTER after min elapsed.
        "max-duration" — max_record_s reached; recording auto-stopped.
    Raises EOFError on closed stdin.

    Why not readline+select? Because readline depends on the TTY driver
    delivering newlines, and audio backends have wedged that in the
    past. ``os.read`` in a background thread is one syscall lower —
    it reads whatever bytes the driver delivers, period — and the hard
    ``max_record_s`` ceiling guarantees the recording stops even if
    everything between us and the keyboard is broken.
    """
    import threading

    saved = _force_canonical_tty()
    enter_event = threading.Event()
    eof_event = threading.Event()
    _start_stdin_watcher(enter_event, eof_event)

    last_heartbeat = time.monotonic()
    try:
        while True:
            time.sleep(WAIT_POLL_S)
            now = time.monotonic()
            elapsed = now - rec_start

            if eof_event.is_set():
                raise EOFError("stdin closed during recording")

            if enter_event.is_set():
                if elapsed >= min_record_s:
                    return "user-stop"
                # Too early — keep recording; the watcher will fire
                # again on the next ENTER.
                print(f"[voice] (only {elapsed:.2f}s — keep talking, "
                      f"another ENTER in ~{min_record_s - elapsed:.1f}s "
                      f"to stop)", flush=True)
                enter_event.clear()
                last_heartbeat = now
                continue

            if elapsed >= max_record_s:
                print(f"[voice] max recording duration "
                      f"({max_record_s:.0f}s) reached — auto-stopping.",
                      flush=True)
                return "max-duration"

            if now - last_heartbeat >= HEARTBEAT_S:
                print(f"[voice] (recording... {elapsed:.1f}s / "
                      f"max {max_record_s:.0f}s — press ENTER to stop)",
                      flush=True)
                last_heartbeat = now
    finally:
        _restore_tty(saved)


def _stop_stream_with_timeout(stream, timeout_s: float) -> None:
    """Call stream.stop() and stream.close() under a timeout.

    PortAudio close has been observed to hang occasionally; we run it on
    a daemon thread and just abandon the descriptor if it doesn't return
    in time. The OS will reclaim the audio device when the process exits.
    """
    import threading
    done = threading.Event()
    def _close():
        try:
            stream.stop()
        except Exception as e:
            print(f"[voice] (stream.stop error: {e!r})", flush=True)
        try:
            stream.close()
        except Exception as e:
            print(f"[voice] (stream.close error: {e!r})", flush=True)
        done.set()
    t = threading.Thread(target=_close, name="audio-close", daemon=True)
    t.start()
    if not done.wait(timeout=timeout_s):
        print(f"[voice] (audio device close hung > {timeout_s}s — "
              "abandoning descriptor)", flush=True)


def _record_via_arecord(min_record_s: float, max_record_s: float,
                        device: Optional[str]) -> bytes:
    """Record using arecord (ALSA CLI) instead of sounddevice/PortAudio.

    Why: PortAudio's ``sd.InputStream`` was hanging indefinitely on the
    laptop — typical causes are a previous instance leaking the audio
    device handle (every script restart inherits the wedge), PulseAudio
    backend hung, or backend negotiation racing with the Booster SDK's
    network init. arecord is a single, well-understood subprocess we
    fully control: it either starts immediately, or fails with a
    diagnosable error within a second.
    """
    import subprocess

    fd, fname = tempfile.mkstemp(suffix=".wav", prefix="ptt_")
    os.close(fd)

    argv = ["arecord", "-q", "-t", "wav",
            "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1"]
    if device:
        argv += ["-D", device]
    argv.append(fname)

    print(f"[voice] spawning: {' '.join(argv)}", flush=True)
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "arecord not found — install alsa-utils: "
            "sudo apt install alsa-utils")

    # Tiny grace period to let arecord either start streaming OR fail
    # fast on a device error. If it has already exited at this point,
    # the audio device is broken and we want a real error, not a wedge.
    time.sleep(0.15)
    if proc.poll() is not None:
        err = proc.stderr.read().decode(errors="replace").strip()
        try:
            os.unlink(fname)
        except OSError:
            pass
        raise RuntimeError(f"arecord exited immediately "
                           f"(rc={proc.returncode}): {err or '(no stderr)'}")

    print(f"[voice] RECORDING (max {max_record_s:.0f}s) — press ENTER "
          f"to stop (min {min_record_s:.1f}s). arecord pid={proc.pid}",
          flush=True)
    rec_start = time.monotonic()
    try:
        reason = _wait_for_stop_enter(min_record_s, max_record_s, rec_start)
        print(f"[voice] recording ended ({reason}) — stopping arecord...",
              flush=True)
    finally:
        # SIGTERM lets arecord flush the WAV header + data properly. If
        # it doesn't shut down in 2s, escalate to SIGKILL — we'd rather
        # truncate the recording than freeze the loop.
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1.0)
        except Exception as e:
            print(f"[voice] (arecord stop error: {e!r})", flush=True)

    try:
        with open(fname, "rb") as f:
            data = f.read()
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass

    # WAV header is typically 44 bytes for canonical PCM; compute audio
    # duration from the file size.
    header_bytes = 44
    sample_bytes = max(0, len(data) - header_bytes)
    duration_s = sample_bytes / (SAMPLE_RATE * 2)   # int16 mono
    print(f"[voice] captured {duration_s:.2f}s "
          f"({sample_bytes} sample bytes, total {len(data)} B).",
          flush=True)

    if duration_s < DISCARD_BELOW_S:
        print(f"[voice] discarding — under {DISCARD_BELOW_S:.1f}s.",
              flush=True)
        return b""
    return data


def _record_via_sounddevice(mic_device, min_record_s: float,
                            max_record_s: float) -> bytes:
    """Original PortAudio path, kept as opt-in fallback (--audio-backend
    sounddevice). Don't use as default — has been observed hanging
    indefinitely on sd.InputStream init.
    """
    import sounddevice as sd
    chunks: list[np.ndarray] = []

    def callback(indata, _frames, _time_info, status):
        if status:
            print(f"[voice] (sd status: {status})", flush=True)
        chunks.append(indata.copy())

    print(f"[voice] opening audio device "
          f"(dev={mic_device!r}, {SAMPLE_RATE} Hz, mono int16)...",
          flush=True)
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                            device=mic_device, callback=callback)
    print("[voice] audio device opened, starting capture...", flush=True)
    stream.start()
    print(f"[voice] RECORDING (max {max_record_s:.0f}s) — press ENTER "
          f"to stop (min {min_record_s:.1f}s).", flush=True)
    rec_start = time.monotonic()
    try:
        reason = _wait_for_stop_enter(min_record_s, max_record_s, rec_start)
        print(f"[voice] recording ended ({reason}) — closing audio device...",
              flush=True)
    finally:
        _stop_stream_with_timeout(stream, STREAM_CLOSE_TIMEOUT_S)

    if not chunks:
        print("[voice] no audio captured.", flush=True)
        return b""
    audio = np.concatenate(chunks, axis=0)
    duration_s = len(audio) / SAMPLE_RATE
    print(f"[voice] captured {duration_s:.2f}s ({len(audio)} samples).",
          flush=True)
    if duration_s < DISCARD_BELOW_S:
        print(f"[voice] discarding — under {DISCARD_BELOW_S:.1f}s.",
              flush=True)
        return b""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def record_push_to_talk(mic_device, max_record_s: float = MAX_RECORD_S,
                        backend: str = "arecord",
                        arecord_device: Optional[str] = None) -> bytes:
    """Push-to-talk recording. Returns 16-bit PCM mono WAV bytes.

    Protocol: ENTER to start, ENTER to stop (or auto-stop at
    ``max_record_s``). KeyboardInterrupt and EOF both propagate so the
    outer voice loop can shut down cleanly.

    ``backend``:
      - ``arecord`` (default): shells out to ALSA's arecord — robust,
        kill-able subprocess. Recommended.
      - ``sounddevice``: original PortAudio path, kept for fallback.
        Has been observed hanging indefinitely on init.
    """
    # Drop any newlines queued before we got here — otherwise the
    # start-ENTER would fire instantly without the user pressing
    # anything.
    _flush_stdin()
    _readline_or_eof("\n[voice] press ENTER to start recording "
                     "(Ctrl-C to exit voice mode)...")

    # Drop newlines the user may have queued while the start prompt was
    # up. If we skip this, the stop-ENTER would fire instantly.
    _flush_stdin()

    if backend == "arecord":
        return _record_via_arecord(MIN_RECORD_S, max_record_s,
                                   device=arecord_device)
    return _record_via_sounddevice(mic_device, MIN_RECORD_S, max_record_s)


# ============================================================================
# Whisper + Groq
# ============================================================================


def transcribe(audio_wav_bytes: bytes, whisper_model) -> str:
    if len(audio_wav_bytes) < 2000:  # <0.06s at 16kHz mono16
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_wav_bytes)
        fname = f.name
    try:
        result = whisper_model.transcribe(fname, fp16=False, language="en")
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass
    return result["text"].strip()


# Quick local "stop" detector — skips Groq for a faster emergency stop.
_STOP_RE = re.compile(r"^(please\s+)?(robot[,\s]+)?stop[\s!.]*$", re.IGNORECASE)


def transcript_to_instruction(transcript: str, groq_client,
                              groq_model: str
                              ) -> Tuple[Optional[str], str]:
    """Map a free-form transcript to a NaVILA instruction (or a chat reply).

    Returns ``(instruction, speech)``:
      - if ``instruction`` is non-None: send it to NaVILA. ``speech`` is the
        TTS line played BEFORE execution (typically the "I heard X" line).
      - if ``instruction`` is None: only speak ``speech`` and skip execution.
    """
    t = transcript.strip()
    if not t:
        return (None, "I didn't catch that.")
    if _STOP_RE.match(t):
        return ("stop", "Stopping.")

    try:
        raw = groq_client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": GROQ_SYS_PROMPT},
                {"role": "user", "content": t},
            ],
            max_tokens=60,
            temperature=0.0,
        ).choices[0].message.content
    except Exception as e:
        print(f"[voice] Groq error: {e!r} — using raw transcript.",
              file=sys.stderr, flush=True)
        return (t, f"I heard: {t}. Executing without cleanup.")

    cmd = clean_groq_output(raw or "")
    if cmd:
        if raw.strip() != cmd:
            print(f"[voice] Groq raw={raw!r}  cleaned={cmd!r}",
                  flush=True)
        return (cmd, f"I heard: {t}. Executing.")
    return (None, "I didn't get a usable command, try again.")


# ============================================================================
# TTS
# ============================================================================


def tts_local(text: str, enable: bool):
    """Speak ``text`` on the local laptop speaker via espeak-ng (non-blocking).
    Always logs the would-be utterance to stdout for visibility.
    """
    clean = text.replace('"', "'").strip()[:400]
    print(f"[voice] >>> {clean}", flush=True)
    if not enable:
        return
    try:
        # stdin=DEVNULL: espeak inherits stdin from us by default; while
        # it doesn't actively read, having a TTY-attached child running
        # alongside our push-to-talk readline has caused terminal-mode
        # weirdness in the past. Explicitly disconnect it from the TTY.
        subprocess.Popen(
            ["espeak-ng", "-s", "165", clean],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("[voice] espeak-ng not found — install with: "
              "sudo apt install espeak-ng", file=sys.stderr, flush=True)


# ============================================================================
# Execute one instruction — wraps the standard relay tick loop
# ============================================================================


def execute_instruction(instruction: str, args, image_src, actuator,
                        stop_requested: threading.Event) -> str:
    """Open a fresh connection to the NaVILA server, send set_instruction,
    run the tick loop until all_done / time-out / network down / SIGINT.

    Returns a short status string.
    """
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

    remote = RemoteVLMClient(
        host=args.server, port=args.port,
        instruction=instruction, cfg=cfg,
        period=args.tick_period, io_timeout=args.io_timeout,
    )
    try:
        remote.connect_and_handshake()
    except Exception as e:
        return f"connect-failed: {e!r}"

    # Bootstrap first frame BEFORE starting the tick thread.
    try:
        first_frame = image_src()
        remote.set_frame_jpeg(encode_jpeg(first_frame, args.jpeg_quality))
    except Exception as e:
        remote.shutdown()
        return f"image-grab-failed: {e!r}"
    last_frame = first_frame

    remote.start_ticking()
    t0 = time.perf_counter()
    next_grab = time.perf_counter()
    next_print = time.perf_counter() + 1.0
    last_grab_ms = 0.0

    # Optional HUD (best-effort; no-op if cv2 missing or --no-display set)
    cv2 = None
    if not args.no_display:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            pass

    try:
        while not stop_requested.is_set():
            now = time.perf_counter()

            if now >= next_grab:
                t_grab = time.perf_counter()
                try:
                    last_frame = image_src()
                    last_grab_ms = (time.perf_counter() - t_grab) * 1000
                    remote.set_frame_jpeg(
                        encode_jpeg(last_frame, args.jpeg_quality))
                except Exception as e:
                    print(f"[voice] image grab failed: {e!r}",
                          file=sys.stderr, flush=True)
                next_grab = now + args.tick_period

            state = remote.get_state()
            if state.connection_lost:
                return f"network-down: {state.last_error}"

            vx = _clip(state.vx, args.vx_max)
            vy = _clip(state.vy, args.vy_max)
            vyaw = _clip(state.vyaw, args.vyaw_max)
            actuator.send(vx, vy, vyaw)

            if state.all_done and state.drain_done:
                return "complete"

            if (now - t0) > args.max_seconds:
                return f"timeout-{args.max_seconds:.0f}s"

            if cv2 is not None:
                link_age = (now - state.last_update_at
                            if state.last_update_at > 0 else float("nan"))
                bgr = draw_hud(last_frame, state, args.mode, instruction,
                               last_grab_ms, link_age, vx, vy, vyaw)
                if bgr is not None:
                    cv2.imshow("NaVILA Voice Relay", bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return "q-key emergency stop"

            if now >= next_print:
                print(f"[voice t={now - t0:5.1f}s] "
                      f"step {state.step_idx + 1}/{state.step_total} "
                      f"[{state.tag}] vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f} "
                      f"raw={state.raw!r}", flush=True)
                next_print = now + 1.0

            time.sleep(max(0.005, 1.0 / max(args.display_hz, 1.0)))
        return "user-stop"
    finally:
        # ALWAYS zero the actuator before returning.
        try:
            actuator.send(0.0, 0.0, 0.0)
        except Exception:
            pass
        try:
            remote.shutdown()
        except Exception:
            pass
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


# ============================================================================
# main
# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- input mode --------------------------------------------------------
    ap.add_argument("--input-mode", choices=["voice", "text"], default="voice",
                    help="voice (push-to-talk loop) or text (one-shot, "
                         "matches the old relay). Default: voice.")
    ap.add_argument("--instruction", default=None,
                    help="--input-mode=text: instruction to send. "
                         "Ignored in voice mode.")

    # ---- voice config ------------------------------------------------------
    ap.add_argument("--whisper-model", default=DEFAULT_WHISPER,
                    choices=WHISPER_CHOICES,
                    help=f"Whisper model name: {'|'.join(WHISPER_CHOICES)}. "
                         f"Default: {DEFAULT_WHISPER}. "
                         "First run downloads ~70MB-3GB.")
    ap.add_argument("--whisper-device", default=DEFAULT_WHISPER_DEVICE,
                    choices=WHISPER_DEVICE_CHOICES,
                    help=f"Where to run Whisper: cpu | cuda | auto. "
                         f"Default: {DEFAULT_WHISPER_DEVICE}. "
                         "CPU is the right choice on the laptop relay — "
                         "the GPU is usually too small for 'medium'. "
                         "'auto' tries CUDA, falls back to CPU on OOM.")
    ap.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS,
                    help=f"Max PyTorch threads for CPU Whisper inference. "
                         f"Default: {DEFAULT_CPU_THREADS}. "
                         "Set lower (e.g. 2) if transcribing makes the "
                         "laptop unresponsive; raise (e.g. 8) for "
                         "faster transcription on workstations.")
    ap.add_argument("--groq-model", default=DEFAULT_GROQ,
                    help="Groq model id for command parsing.")
    ap.add_argument("--no-groq-cleanup", action="store_true",
                    help="Send the raw Whisper transcript to NaVILA instead "
                         "of routing through Groq. Useful when offline.")
    ap.add_argument("--mic-device", default=None,
                    help="sounddevice index or substring. "
                         "Pass --list-mics to enumerate.")
    ap.add_argument("--list-mics", action="store_true",
                    help="Print available input devices and exit.")
    ap.add_argument("--no-tts", action="store_true",
                    help="Suppress espeak-ng feedback (still printed).")
    ap.add_argument("--max-record-seconds", type=float, default=MAX_RECORD_S,
                    help=f"Hard upper bound on a single push-to-talk "
                         f"recording. Default: {MAX_RECORD_S:.0f}s. "
                         "Recording auto-stops at this point even if no "
                         "ENTER is detected — guarantees the loop can't "
                         "wedge forever.")
    ap.add_argument("--audio-backend", choices=("arecord", "sounddevice"),
                    default="arecord",
                    help="How to capture audio. 'arecord' (default) "
                         "shells out to the ALSA CLI — robust and "
                         "kill-able. 'sounddevice' uses PortAudio "
                         "via the sounddevice library; has been "
                         "observed hanging on InputStream init when "
                         "the audio device is busy / leaked.")
    ap.add_argument("--arecord-device", default=None,
                    help="ALSA device for arecord (e.g. 'default', "
                         "'plughw:1,0'). Default: arecord's default.")

    # ---- remote server -----------------------------------------------------
    ap.add_argument("--server", help="Desktop NaVILA server host/IP "
                                     "(Tailscale).")
    ap.add_argument("--port", type=int, default=proto.DEFAULT_PORT)
    ap.add_argument("--io-timeout", type=float, default=5.0)

    # ---- actuator (carried over from navila_laptop_relay) ------------------
    ap.add_argument("--mode", choices=["print", "dry", "live"], default="print")
    ap.add_argument("--net", type=str, default="192.168.10.102")
    ap.add_argument("--send-hz", type=float, default=20.0)
    ap.add_argument("--watchdog-seconds", type=float, default=1.5)

    # ---- image source ------------------------------------------------------
    ap.add_argument("--image-source",
                    choices=["zed", "mjpeg", "dir", "static"],
                    default="static")
    ap.add_argument("--image-path", type=Path,
                    default=Path.home() /
                            "Projects/k1_research/experiments/vla/test_image.jpg")
    ap.add_argument("--image-dir", type=Path)
    ap.add_argument("--mjpeg-url", type=str,
                    default="http://192.168.10.102:8080/stream")
    ap.add_argument("--zed-resolution", default="VGA")
    ap.add_argument("--zed-fps", type=int, default=30)
    ap.add_argument("--jpeg-quality", type=int, default=80)

    # ---- planner caps + cadence -------------------------------------------
    ap.add_argument("--per-step-time", type=float, default=25.0)
    ap.add_argument("--proximity-threshold", type=float, default=1.0)
    ap.add_argument("--action-duration", type=float, default=ACTION_DURATION)
    ap.add_argument("--vx-max", type=float, default=0.4)
    ap.add_argument("--vy-max", type=float, default=0.15)
    ap.add_argument("--vyaw-max", type=float, default=0.4)
    ap.add_argument("--heading-assist",
                    action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--turn-controller",
                    action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--tick-period", type=float, default=0.4)
    ap.add_argument("--display-hz", type=float, default=20.0)
    ap.add_argument("--max-seconds", type=float, default=300.0,
                    help="Per-instruction time limit. Voice loop continues "
                         "after each instruction.")
    ap.add_argument("--no-display", action="store_true",
                    help="Disable the cv2 HUD window.")

    args = ap.parse_args()

    # --list-mics short-circuit before pulling in any of the relay machinery.
    if args.list_mics:
        try:
            print(list_input_devices())
        except Exception as e:
            print(f"[voice] list-mics failed: {e!r}", file=sys.stderr)
            sys.exit(2)
        return

    if not args.server:
        ap.error("--server is required (the desktop NaVILA host/IP).")

    # Resolve mic-device: int if all-digits, else pass through (string match).
    mic_device = args.mic_device
    if mic_device is not None:
        try:
            mic_device = int(mic_device)
        except ValueError:
            pass

    # Shared state for SIGINT.
    stop_requested = threading.Event()

    def _sigint(_signum, _frame):
        if not stop_requested.is_set():
            print("\n[voice] SIGINT — emergency stopping current instruction.",
                  flush=True)
        stop_requested.set()

    signal.signal(signal.SIGINT, _sigint)

    # ----------- text mode = one-shot (old relay behavior) ------------------
    if args.input_mode == "text":
        if not args.instruction:
            ap.error("--input-mode=text requires --instruction.")
        image_src = make_image_source(args)
        actuator = make_actuator(args)
        actuator.init()
        tts_local(f"Executing: {args.instruction}", enable=not args.no_tts)
        try:
            result = execute_instruction(args.instruction, args, image_src,
                                         actuator, stop_requested)
        finally:
            try:
                actuator.shutdown()
            except Exception:
                pass
            try:
                image_src.close()
            except Exception:
                pass
        tts_local(f"Done: {result}", enable=not args.no_tts)
        return

    # ----------- voice mode ------------------------------------------------
    import whisper
    import torch

    # Cap PyTorch CPU threads BEFORE loading Whisper. Without this,
    # PyTorch defaults to using every logical CPU, and a single Whisper
    # transcribe call pins the entire system unresponsive for as long as
    # it runs. Picking min(--cpu-threads, ncpu//2 or 1) leaves at least
    # half the cores free for the rest of the OS while still giving
    # Whisper enough parallelism to run quickly.
    ncpu = os.cpu_count() or 4
    cpu_threads = max(1, min(args.cpu_threads, max(1, ncpu // 2 + 1)))
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(min(2, cpu_threads))
    except RuntimeError:
        # set_num_interop_threads can only be called before any parallel
        # work has started; ignore if someone else already touched it.
        pass
    print(f"[voice] PyTorch CPU threads capped at {cpu_threads} "
          f"(detected {ncpu} logical cores) to keep the system "
          f"responsive during transcribe.", flush=True)

    def _load_whisper(name: str, device: str):
        if device == "auto":
            try:
                print(f"[voice] loading Whisper ({name}) on CUDA...",
                      flush=True)
                return whisper.load_model(name, device="cuda")
            except Exception as e:
                msg = str(e).lower()
                if "out of memory" in msg or "cuda" in msg:
                    print(f"[voice] CUDA load failed ({type(e).__name__}: "
                          f"{e}); falling back to CPU.",
                          file=sys.stderr, flush=True)
                    return whisper.load_model(name, device="cpu")
                raise
        print(f"[voice] loading Whisper ({name}) on {device}...",
              flush=True)
        return whisper.load_model(name, device=device)

    whisper_model = _load_whisper(args.whisper_model, args.whisper_device)
    print("[voice] Whisper ready.", flush=True)

    if not args.no_groq_cleanup:
        try:
            from groq import Groq
        except ImportError:
            print("[voice] groq package missing — install or pass "
                  "--no-groq-cleanup.", file=sys.stderr)
            sys.exit(2)
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("[voice] GROQ_API_KEY env var not set. "
                  "export it or pass --no-groq-cleanup.", file=sys.stderr)
            sys.exit(2)
        groq_client = Groq(api_key=api_key)
    else:
        groq_client = None

    # Image source + actuator stay open for the whole voice loop.
    image_src = make_image_source(args)
    actuator = make_actuator(args)
    actuator.init()

    tts_local("Voice control ready. Press enter to speak.",
              enable=not args.no_tts)

    try:
        while True:
            stop_requested.clear()  # fresh per instruction
            try:
                audio = record_push_to_talk(
                    mic_device,
                    max_record_s=args.max_record_seconds,
                    backend=args.audio_backend,
                    arecord_device=args.arecord_device,
                )
            except (EOFError, KeyboardInterrupt):
                print("\n[voice] stopping (Ctrl-C / EOF).", flush=True)
                break

            transcript = transcribe(audio, whisper_model)
            if not transcript:
                tts_local("I didn't catch that.", enable=not args.no_tts)
                continue
            print(f"[voice] transcript: {transcript!r}", flush=True)

            if args.no_groq_cleanup:
                instruction = transcript
                tts_local(f"I heard: {transcript}. Executing.",
                          enable=not args.no_tts)
            else:
                instruction, speech = transcript_to_instruction(
                    transcript, groq_client, args.groq_model)
                tts_local(speech, enable=not args.no_tts)
                if instruction is None:
                    continue

            print(f"[voice] → NaVILA set_instruction: {instruction!r}",
                  flush=True)
            try:
                result = execute_instruction(
                    instruction, args, image_src, actuator, stop_requested)
            except Exception as e:
                print(f"[voice] execute_instruction crashed: {e!r}",
                      file=sys.stderr, flush=True)
                tts_local(f"Error: {e}", enable=not args.no_tts)
                continue

            print(f"[voice] result: {result}", flush=True)
            tts_local(
                "Task complete." if result == "complete"
                else f"Stopped: {result.split(':')[0]}",
                enable=not args.no_tts,
            )
    except KeyboardInterrupt:
        print("\n[voice] outer Ctrl-C.", flush=True)
    finally:
        try:
            actuator.send(0.0, 0.0, 0.0)
            actuator.shutdown()
        except Exception:
            pass
        try:
            image_src.close()
        except Exception:
            pass
        tts_local("Voice control off.", enable=not args.no_tts)
        print("[voice] bye.", flush=True)


if __name__ == "__main__":
    main()
