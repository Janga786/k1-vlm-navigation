#!/usr/bin/env python3
"""Standalone test for the voice capture + Whisper transcription path.

Verifies that the mic + sounddevice + Whisper stack works on this machine
BEFORE you involve the NaVILA server. Run this once on a fresh machine
(laptop) and confirm the transcript matches what you said.

Usage::

    # List available input devices
    python test_voice_capture.py --list-mics

    # Record 5 seconds from the default mic, transcribe, print
    python test_voice_capture.py

    # Pick a specific mic by index or name substring
    python test_voice_capture.py --mic-device 2
    python test_voice_capture.py --mic-device "USB Audio"

    # Pick a different Whisper size
    python test_voice_capture.py --whisper-model tiny

The script uses the same record_push_to_talk() / transcribe() helpers as
``navila_voice_relay.py`` — if the test passes here, voice mode will work
in the relay.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from navila_voice_relay import (
    DEFAULT_WHISPER,
    list_input_devices,
    record_push_to_talk,
    transcribe,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-mics", action="store_true")
    ap.add_argument("--mic-device", default=None)
    ap.add_argument("--whisper-model", default=DEFAULT_WHISPER)
    args = ap.parse_args()

    if args.list_mics:
        print(list_input_devices())
        return

    # Resolve mic device (int if all-digit, else string)
    mic_device = args.mic_device
    if mic_device is not None:
        try:
            mic_device = int(mic_device)
        except ValueError:
            pass

    print(f"[test] loading Whisper '{args.whisper_model}' "
          "(first run downloads weights)...")
    import whisper
    model = whisper.load_model(args.whisper_model)
    print("[test] Whisper ready.")

    try:
        wav_bytes = record_push_to_talk(mic_device)
    except (KeyboardInterrupt, EOFError):
        print("\n[test] aborted.")
        return

    if len(wav_bytes) < 2000:
        print("[test] no audio captured. Check your mic.")
        return

    print(f"[test] {len(wav_bytes)} WAV bytes captured. Transcribing...")
    text = transcribe(wav_bytes, model)
    if not text:
        print("[test] Whisper returned empty. Mic level too low?")
        return
    print(f"\n[test] TRANSCRIPT: {text!r}\n")
    print("[test] OK — voice capture pipeline works on this machine.")


if __name__ == "__main__":
    main()
