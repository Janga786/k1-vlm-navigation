"""Tests that VLMRunner.reset_for_session() fully clears per-instruction
state — including in-flight inference results that complete after the
reset (the "I think I should stop" on instruction B bug).

No torch / no llava / no GPU — uses a fake inference function injected
via VLMRunner.set_inference_fn so the loop runs against pure Python.
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navila_k1_core import VLMRunner  # noqa: E402


def _img() -> Image.Image:
    return Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))


class ResetForSessionTests(unittest.TestCase):

    def test_clears_buffer_stop_cmd_label(self):
        v = VLMRunner(model_path=Path("/nope"))
        # populate state
        v._frame_buffer.append(_img())  # noqa: SLF001
        v._frame_buffer.append(_img())  # noqa: SLF001
        v.stop_event.set()
        v._publish(0.5, 0.0, 0.0, "forward", "raw", 100.0)
        self.assertEqual(len(v._frame_buffer), 2)  # noqa: SLF001
        self.assertTrue(v.stop_event.is_set())
        self.assertEqual(v.get_command(), (0.5, 0.0, 0.0))

        v.reset_for_session()

        self.assertEqual(len(v._frame_buffer), 0)  # noqa: SLF001
        self.assertFalse(v.stop_event.is_set())
        self.assertEqual(v.get_command(), (0.0, 0.0, 0.0))
        self.assertIn("cleared", v.status()["label"])
        self.assertEqual(v.status()["raw"], "")
        self.assertEqual(v.status()["inf_count"], 0)

    def test_gen_id_bumps_each_reset(self):
        v = VLMRunner(model_path=Path("/nope"))
        g0 = v._current_gen()  # noqa: SLF001
        v.reset_for_session()
        g1 = v._current_gen()  # noqa: SLF001
        v.reset_for_session()
        g2 = v._current_gen()  # noqa: SLF001
        self.assertEqual(g1, g0 + 1)
        self.assertEqual(g2, g0 + 2)


class PublishGatingTests(unittest.TestCase):
    """The core fix: an in-flight inference (started before the reset)
    must NOT clobber the new instruction's state when it finishes."""

    def test_publish_dropped_when_gen_changed(self):
        v = VLMRunner(model_path=Path("/nope"))
        gen_at_start = v._current_gen()  # noqa: SLF001

        # Simulate: reset happens while inference is in flight.
        v.reset_for_session()

        # Late publish from the previous instruction tries to land.
        landed = v._publish(0.5, 0.0, 0.1, "forward", "stale raw",
                            123.0, gen_at_start=gen_at_start)
        self.assertFalse(landed, "stale publish must be dropped")
        # State must remain in the freshly-reset shape.
        self.assertEqual(v.get_command(), (0.0, 0.0, 0.0))
        self.assertEqual(v.status()["raw"], "")
        self.assertEqual(v.status()["inf_count"], 0)

    def test_publish_lands_when_gen_unchanged(self):
        v = VLMRunner(model_path=Path("/nope"))
        v.reset_for_session()
        gen_at_start = v._current_gen()  # noqa: SLF001

        landed = v._publish(0.4, 0.0, 0.0, "forward", "ok",
                            50.0, gen_at_start=gen_at_start)
        self.assertTrue(landed)
        self.assertEqual(v.get_command(), (0.4, 0.0, 0.0))
        self.assertEqual(v.status()["raw"], "ok")
        self.assertEqual(v.status()["inf_count"], 1)


class InferenceLoopGatingTests(unittest.TestCase):
    """End-to-end test of the inference loop: inject a slow fake
    inference function, trigger reset_for_session() while it's running,
    and assert the result that comes out AFTER the reset doesn't touch
    the new instruction's state."""

    def test_in_flight_result_dropped_across_reset(self):
        v = VLMRunner(model_path=Path("/nope"))

        # Fake inference: per-call returns. Call 0 returns "stop"
        # (the stale answer that must be DROPPED). Call 1+ returns
        # "move forward 25 cm" (the legitimate answer for instruction B
        # that MUST land). Both calls block on `releases[idx]` so the
        # test can deterministically order: start call 0, reset, set
        # instruction B, release call 0 (gets dropped), let call 1
        # proceed naturally.
        releases = [threading.Event(), threading.Event()]
        returns = ["stop", "move forward 25 cm"]
        calls_seen = []   # list[(idx, instruction)]
        lock = threading.Lock()

        def fake(frames, instruction):
            with lock:
                idx = len(calls_seen)
                calls_seen.append((idx, instruction))
            releases[idx].wait(timeout=5.0)
            return returns[idx]

        v.set_inference_fn(fake)
        v._frame_buffer.append(_img())  # noqa: SLF001
        v.set_instruction("instruction A")
        v.start()

        # Wait for the loop to enter inference call #0 (under A).
        for _ in range(200):
            with lock:
                started = len(calls_seen) >= 1
            if started:
                break
            time.sleep(0.02)
        self.assertEqual(calls_seen[0], (0, "instruction A"))

        # Client B reconnects: reset + new instruction. Gen bumps 0→1.
        v.reset_for_session()
        v.set_instruction("instruction B")
        v._frame_buffer.append(_img())  # noqa: SLF001

        # Release the stale call from A. It returns "stop" but its
        # captured gen (0) no longer matches current gen (1) → drop.
        releases[0].set()

        # Wait for the loop to enter inference call #1 (under B).
        for _ in range(200):
            with lock:
                next_started = len(calls_seen) >= 2
            if next_started:
                break
            time.sleep(0.02)
        self.assertEqual(calls_seen[1], (1, "instruction B"),
                          "second inference must run under instruction B")

        # IMPORTANT assertion #1: between release of A and start of B,
        # stop_event must NOT have been set by the dropped A result.
        self.assertFalse(v.stop_event.is_set(),
                          "stale 'stop' from instruction A leaked into "
                          "instruction B's stop_event — gating failed")
        # And cmd is still the post-reset zero — A's stale publish did
        # not clobber it.
        self.assertEqual(v.get_command(), (0.0, 0.0, 0.0))
        self.assertEqual(v.status()["inf_count"], 0,
                          "stale publish should not have incremented "
                          "the per-session inference counter")

        # Now release B's call — it returns "move forward" and SHOULD
        # publish. Confirms the gate doesn't accidentally block
        # legitimate gen-matched publishes.
        releases[1].set()
        for _ in range(200):
            if v.get_command()[0] > 0.0:
                break
            time.sleep(0.02)
        self.assertGreater(v.get_command()[0], 0.0,
                            "instruction B's legit publish did not land")
        self.assertEqual(v.status()["inf_count"], 1,
                          "instruction B's legit publish should bump "
                          "the (reset-to-zero) inference counter to 1")

        v.shutdown(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
