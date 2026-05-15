"""Unit tests for the NaVILA → (vx, vy, vyaw, duration) parser.

No deps beyond stdlib + the bridge module (which doesn't import torch
unless its main() runs). These should pass on any Python 3.10+.

NaVILA paper §II-B mandates fixed command velocities (0.5 m/s for forward,
±π/6 rad/s for turns, 0 for stop) with the time held proportional to the
requested distance / angle. parse_action() implements that mapping.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navila_k1_bridge import (  # noqa: E402
    FORWARD_SPEED, TURN_SPEED, parse_action,
)


class ParseActionTests(unittest.TestCase):

    DURATION = 1.5  # legacy fallback; only used for unparsed text

    # --- forward / backward -------------------------------------------------

    def test_forward_cm(self):
        vx, vy, vyaw, dur, label = parse_action("move forward 75 cm",
                                                 self.DURATION)
        self.assertAlmostEqual(vx, FORWARD_SPEED, places=6)
        self.assertEqual(vy, 0.0)
        self.assertEqual(vyaw, 0.0)
        self.assertAlmostEqual(dur, 0.75 / FORWARD_SPEED, places=6)
        self.assertIn("forward", label)

    def test_forward_meters(self):
        vx, _, _, dur, label = parse_action("walk forward 0.5 m", self.DURATION)
        self.assertAlmostEqual(vx, FORWARD_SPEED, places=6)
        self.assertAlmostEqual(dur, 0.5 / FORWARD_SPEED, places=6)
        self.assertIn("forward", label)

    def test_backward(self):
        vx, _, _, dur, label = parse_action("move back 30 cm", self.DURATION)
        self.assertAlmostEqual(vx, -FORWARD_SPEED, places=6)
        self.assertAlmostEqual(dur, 0.30 / FORWARD_SPEED, places=6)
        self.assertIn("backward", label)

    def test_forward_with_extra_phrasing(self):
        # NaVILA usually wraps the answer with prose; parser must still find
        # the action.
        vx, _, _, dur, _ = parse_action(
            "The next action is move forward 25 cm.", self.DURATION)
        self.assertAlmostEqual(vx, FORWARD_SPEED, places=6)
        self.assertAlmostEqual(dur, 0.25 / FORWARD_SPEED, places=6)  # 0.5 s

    # --- turns --------------------------------------------------------------

    def test_turn_left_deg(self):
        _, _, vyaw, dur, label = parse_action("turn left 30 degrees",
                                               self.DURATION)
        self.assertAlmostEqual(vyaw, TURN_SPEED, places=6)
        self.assertAlmostEqual(dur, math.radians(30) / TURN_SPEED, places=6)
        self.assertIn("left", label.lower())

    def test_turn_right_deg(self):
        _, _, vyaw, dur, label = parse_action("turn right 90 degrees",
                                               self.DURATION)
        self.assertAlmostEqual(vyaw, -TURN_SPEED, places=6)
        self.assertAlmostEqual(dur, math.radians(90) / TURN_SPEED, places=6)
        self.assertIn("right", label.lower())

    def test_turn_radians(self):
        _, _, vyaw, dur, _ = parse_action("turn left 0.5 rad", self.DURATION)
        self.assertAlmostEqual(vyaw, TURN_SPEED, places=6)
        self.assertAlmostEqual(dur, 0.5 / TURN_SPEED, places=6)

    def test_turn_speed_is_always_pi_over_six(self):
        # Whatever the angle, the commanded speed is exactly π/6 rad/s
        # (paper-spec). Duration is what scales.
        for n_deg in (10, 30, 45, 90, 180):
            _, _, vyaw, dur, _ = parse_action(
                f"turn left {n_deg} degrees", self.DURATION)
            self.assertAlmostEqual(vyaw, TURN_SPEED, places=6,
                                    msg=f"turn left {n_deg}° vyaw")
            self.assertAlmostEqual(
                dur, math.radians(n_deg) / TURN_SPEED, places=6,
                msg=f"turn left {n_deg}° duration")

    # --- stop / unparseable -------------------------------------------------

    def test_explicit_stop(self):
        vx, vy, vyaw, dur, label = parse_action("stop", self.DURATION)
        self.assertEqual((vx, vy, vyaw), (0.0, 0.0, 0.0))
        self.assertEqual(dur, 0.0)
        self.assertEqual(label, "stop")

    def test_completed_synonym(self):
        _, _, _, _, label = parse_action(
            "The task is completed.", self.DURATION)
        self.assertEqual(label, "stop")

    def test_garbage_falls_through_to_stop(self):
        vx, vy, vyaw, dur, label = parse_action(
            "I see a chair.", self.DURATION)
        self.assertEqual((vx, vy, vyaw), (0.0, 0.0, 0.0))
        self.assertEqual(dur, 0.0)
        self.assertNotEqual(label, "stop")
        self.assertIn("unparsed", label)

    def test_empty_string(self):
        vx, vy, vyaw, dur, label = parse_action("", self.DURATION)
        self.assertEqual((vx, vy, vyaw), (0.0, 0.0, 0.0))
        self.assertEqual(dur, 0.0)
        self.assertIn("unparsed", label)

    # --- paper-spec invariants ---------------------------------------------

    def test_forward_speed_is_always_half_mps(self):
        for cm in (10, 25, 50, 75, 100, 250):
            vx, _, _, dur, _ = parse_action(
                f"move forward {cm} cm", self.DURATION)
            self.assertAlmostEqual(vx, FORWARD_SPEED, places=6,
                                    msg=f"forward {cm}cm speed")
            self.assertAlmostEqual(dur, (cm / 100.0) / FORWARD_SPEED, places=6,
                                    msg=f"forward {cm}cm duration")


if __name__ == "__main__":
    unittest.main()
