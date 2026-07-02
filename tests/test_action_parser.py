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
        self.assertLessEqual(abs(vyaw), 0.41)  # bridge.VYAW_MAX = 0.4 (deploy cap)

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


class ParseActionHoldTests(unittest.TestCase):
    """Execute-to-completion hold: speed x hold must equal the commanded
    displacement even when velocity was clipped by the caps (the old code
    held a clipped turn for a fixed 1.5s -> 'turn right 90' executed ~15deg)."""

    DURATION = 1.5

    def test_turn_90_completes_the_angle(self):
        from navila_k1_bridge import parse_action_hold
        _, _, vyaw, _, hold = parse_action_hold("turn right 90 degrees",
                                                self.DURATION)
        # 1.571 rad at |vyaw|<=0.4 -> hold ~3.93s; angle covered ~90deg.
        self.assertAlmostEqual(abs(vyaw) * hold, math.radians(90), places=2)

    def test_unclipped_turn_keeps_default_duration(self):
        from navila_k1_bridge import parse_action_hold
        _, _, vyaw, _, hold = parse_action_hold("turn left 30 degrees",
                                                self.DURATION)
        # 0.524/1.5 = 0.349 < 0.4 cap -> no clip -> hold == duration.
        self.assertAlmostEqual(hold, self.DURATION, places=3)
        self.assertAlmostEqual(abs(vyaw) * hold, math.radians(30), places=2)

    def test_forward_clipped_completes_the_distance(self):
        from navila_k1_bridge import parse_action_hold
        vx, _, _, _, hold = parse_action_hold("move forward 75 cm",
                                              self.DURATION)
        # 0.75/1.5=0.5 -> clipped to 0.4 -> hold stretches to 1.875s.
        self.assertAlmostEqual(vx * hold, 0.75, places=3)

    def test_hold_capped(self):
        from navila_k1_bridge import HOLD_MAX, parse_action_hold
        _, _, _, _, hold = parse_action_hold("turn left 720 degrees",
                                             self.DURATION)
        self.assertLessEqual(hold, HOLD_MAX + 1e-9)

    def test_stop_and_unparsed_have_zero_hold(self):
        from navila_k1_bridge import parse_action_hold
        for text in ("stop", "I see a chair."):
            *_, hold = parse_action_hold(text, self.DURATION)
            self.assertEqual(hold, 0.0)


if __name__ == "__main__":
    unittest.main()
