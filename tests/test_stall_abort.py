"""Windowed-progress stall-abort backstop in check_termination.

Thresholds come from the validated benchmark hopeless-stall gate
(2026-06-10, L1.4/L2b): a true wedge measures ~0.13 m / 8 deg net per 10 s
window and must FIRE; an escaping grind (~0.50 m / 37 deg) and ordinary
walking/turning must NOT. A wedged biped wobbles (~1 rad/s base noise) and
creeps (~0.05 m/s), so the gate uses NET windowed progress, never
instantaneous velocity.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navila_k1_core import (  # noqa: E402
    SubStep, TerminationState, check_termination,
)

WINDOW = 10.0


def _run(pos_fn, yaw_fn, cmd_moving=True, t_end=25.0, dt=0.1,
         time_limit=60.0):
    """Drive check_termination through simulated time; return (reason, t)."""
    ss = SubStep(instruction="walk to the wall", time_limit=time_limit)
    state = TerminationState(step_idx=0, started_at=0.0,
                             start_yaw=0.0, last_yaw=0.0)
    t = 0.0
    while t <= t_end:
        state.yaw_unwrap = yaw_fn(t)
        reason = check_termination(
            ss, state,
            current_pos_xy=pos_fn(t),
            vlm_stop=False,
            now=t,
            cmd_moving=cmd_moving,
            stall_window_s=WINDOW,
        )
        if reason is not None:
            return reason, t
        t += dt
    return None, t


class StallAbortTests(unittest.TestCase):

    def test_true_wedge_fires_at_window(self):
        # L1.4-measured wedge: ~0.013 m/s creep, ~0.8 deg/s wobble drift.
        reason, t = _run(pos_fn=lambda t: (0.013 * t, 0.0),
                         yaw_fn=lambda t: math.radians(0.8) * t)
        self.assertIsNotNone(reason)
        self.assertIn("stall-abort", reason)
        self.assertAlmostEqual(t, WINDOW, delta=0.5)

    def test_normal_walk_does_not_fire(self):
        # 0.4 m/s forward: window re-anchors on progress, never fires.
        reason, _ = _run(pos_fn=lambda t: (0.4 * t, 0.0),
                         yaw_fn=lambda t: 0.0)
        self.assertIsNone(reason)

    def test_turn_in_place_does_not_fire(self):
        # 0.4 rad/s turn: zero displacement but yaw progress re-anchors.
        reason, _ = _run(pos_fn=lambda t: (0.0, 0.0),
                         yaw_fn=lambda t: 0.4 * t)
        self.assertIsNone(reason)

    def test_escape_grind_does_not_fire(self):
        # The benchmark's ep0 grind-slide regime: 0.05 m/s + 3.7 deg/s —
        # still escaping, must be allowed to continue.
        reason, _ = _run(pos_fn=lambda t: (0.05 * t, 0.0),
                         yaw_fn=lambda t: math.radians(3.7) * t)
        self.assertIsNone(reason)

    def test_standing_with_zero_cmd_never_fires(self):
        # cmd_moving False = the robot was TOLD to stand — not a stall.
        reason, _ = _run(pos_fn=lambda t: (0.0, 0.0),
                         yaw_fn=lambda t: 0.0,
                         cmd_moving=False)
        self.assertIsNone(reason)

    def test_no_pose_never_fires(self):
        ss = SubStep(instruction="walk", time_limit=60.0)
        state = TerminationState(step_idx=0, started_at=0.0,
                                 start_yaw=0.0, last_yaw=0.0)
        for t in range(0, 30):
            reason = check_termination(
                ss, state, current_pos_xy=None, vlm_stop=False,
                now=float(t), cmd_moving=True, stall_window_s=WINDOW)
            self.assertIsNone(reason)

    def test_anchor_reset_between_substeps(self):
        # A wedge accumulated in one sub-step must not fire instantly in the
        # next one after reset_stall_anchor().
        ss = SubStep(instruction="walk", time_limit=60.0)
        state = TerminationState(step_idx=0, started_at=0.0,
                                 start_yaw=0.0, last_yaw=0.0)
        for t in [0.0, 5.0, 9.5]:   # anchor set, nearly ripe
            check_termination(ss, state, current_pos_xy=(0.0, 0.0),
                              vlm_stop=False, now=t, cmd_moving=True,
                              stall_window_s=WINDOW)
        state.reset_stall_anchor()
        state.started_at = 10.0
        reason = check_termination(ss, state, current_pos_xy=(0.0, 0.0),
                                   vlm_stop=False, now=10.5, cmd_moving=True,
                                   stall_window_s=WINDOW)
        self.assertIsNone(reason)   # fresh window, not ripe yet

    def test_intermittent_cmd_resets_window(self):
        # Motion command pauses mid-window (VLM said stop/zero-hold): the
        # anchor must clear so the stall clock doesn't count standing time.
        ss = SubStep(instruction="walk", time_limit=60.0)
        state = TerminationState(step_idx=0, started_at=0.0,
                                 start_yaw=0.0, last_yaw=0.0)
        # 8s of wedged motion (not yet ripe) ...
        for t in [0.0, 4.0, 8.0]:
            self.assertIsNone(check_termination(
                ss, state, current_pos_xy=(0.0, 0.0), vlm_stop=False,
                now=t, cmd_moving=True, stall_window_s=WINDOW))
        # ... 4s of commanded standstill clears the anchor ...
        self.assertIsNone(check_termination(
            ss, state, current_pos_xy=(0.0, 0.0), vlm_stop=False,
            now=12.0, cmd_moving=False, stall_window_s=WINDOW))
        # ... resumed motion starts a FRESH window (fires at ~12+10, not 10).
        self.assertIsNone(check_termination(
            ss, state, current_pos_xy=(0.0, 0.0), vlm_stop=False,
            now=13.0, cmd_moving=True, stall_window_s=WINDOW))
        self.assertIsNone(check_termination(
            ss, state, current_pos_xy=(0.0, 0.0), vlm_stop=False,
            now=20.0, cmd_moving=True, stall_window_s=WINDOW))
        reason = check_termination(
            ss, state, current_pos_xy=(0.0, 0.0), vlm_stop=False,
            now=23.5, cmd_moving=True, stall_window_s=WINDOW)
        self.assertIsNotNone(reason)
        self.assertIn("stall-abort", reason)


if __name__ == "__main__":
    unittest.main()
