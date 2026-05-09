"""Unit tests for the NaVILA → (vx, vy, vyaw) parser.

No deps beyond stdlib + the bridge module (which doesn't import torch
unless its main() runs). These should pass on any Python 3.10+.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navila_k1_bridge import parse_action  # noqa: E402


class ParseActionTests(unittest.TestCase):

    DURATION = 1.5  # ACTION_DURATION default; tests use it explicitly

    # --- forward / backward -------------------------------------------------

    def test_forward_cm(self):
        vx, vy, vyaw, label = parse_action("move forward 75 cm", self.DURATION)
        self.assertAlmostEqual(vx, 0.4, places=2)  # 0.75/1.5 capped at VX_MAX=0.4
        self.assertEqual(vy, 0.0)
        self.assertEqual(vyaw, 0.0)
        self.assertIn("forward", label)

    def test_forward_meters(self):
        vx, _, _, label = parse_action("walk forward 0.5 m", self.DURATION)
        self.assertAlmostEqual(vx, 0.5 / self.DURATION, places=3)
        self.assertIn("forward", label)

    def test_backward(self):
        vx, _, _, label = parse_action("move back 30 cm", self.DURATION)
        self.assertLess(vx, 0.0)
        self.assertIn("backward", label)

    def test_forward_with_extra_phrasing(self):
        # NaVILA usually wraps the answer with prose; parser must still find
        # the action.
        vx, _, _, _ = parse_action(
            "The next action is move forward 25 cm.", self.DURATION)
        self.assertGreater(vx, 0.0)

    # --- turns --------------------------------------------------------------

    def test_turn_left_deg(self):
        _, _, vyaw, label = parse_action("turn left 30 degrees", self.DURATION)
        self.assertGreater(vyaw, 0.0)         # left = positive
        self.assertIn("left", label.lower())

    def test_turn_right_deg(self):
        _, _, vyaw, label = parse_action("turn right 90 degrees", self.DURATION)
        self.assertLess(vyaw, 0.0)            # right = negative
        self.assertIn("right", label.lower())

    def test_turn_radians(self):
        _, _, vyaw, _ = parse_action("turn left 0.5 rad", self.DURATION)
        self.assertGreater(vyaw, 0.0)

    def test_turn_clipped(self):
        # 180° / 1.5s = 2.09 rad/s, well above VYAW_MAX. Should clip.
        _, _, vyaw, _ = parse_action("turn left 180 degrees", self.DURATION)
        self.assertLessEqual(abs(vyaw), 0.21)  # bridge.VYAW_MAX = 0.2

    # --- stop / unparseable -------------------------------------------------

    def test_explicit_stop(self):
        vx, vy, vyaw, label = parse_action("stop", self.DURATION)
        self.assertEqual((vx, vy, vyaw), (0.0, 0.0, 0.0))
        self.assertEqual(label, "stop")

    def test_completed_synonym(self):
        _, _, _, label = parse_action(
            "The task is completed.", self.DURATION)
        self.assertEqual(label, "stop")

    def test_garbage_falls_through_to_stop(self):
        vx, vy, vyaw, label = parse_action(
            "I see a chair.", self.DURATION)
        self.assertEqual((vx, vy, vyaw), (0.0, 0.0, 0.0))
        # Distinguish "intentional stop" from "unparsed".
        self.assertNotEqual(label, "stop")
        self.assertIn("unparsed", label)

    def test_empty_string(self):
        vx, vy, vyaw, label = parse_action("", self.DURATION)
        self.assertEqual((vx, vy, vyaw), (0.0, 0.0, 0.0))
        self.assertIn("unparsed", label)

    # --- duration scaling ---------------------------------------------------

    def test_shorter_duration_yields_higher_speed(self):
        v1, *_ = parse_action("move forward 30 cm", duration=1.5)
        v2, *_ = parse_action("move forward 30 cm", duration=0.5)
        # Both should be positive, v2 > v1 (subject to clipping).
        self.assertGreater(v2, v1)
        self.assertGreater(v1, 0.0)


if __name__ == "__main__":
    unittest.main()
