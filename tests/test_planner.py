"""Unit tests for the multi-step planner: parser + termination logic.

No NaVILA, no MuJoCo, no SDK. Pure-Python tests on the core module.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navila_k1_core import (  # noqa: E402
    DEFAULT_SCENE_TARGETS,
    SubStep,
    TerminationState,
    apply_controllers,
    check_termination,
    parse_substeps,
    update_yaw_unwrap,
    wrap_pi,
    yaw_from_quat,
)


# ===========================================================================
# Sub-step parsing
# ===========================================================================


class ParseSubstepsTests(unittest.TestCase):

    def setUp(self):
        self.targets = DEFAULT_SCENE_TARGETS
        self.t = 25.0
        self.thr = 1.0

    def test_single_step(self):
        ss = parse_substeps("walk forward", self.targets, self.t, self.thr)
        self.assertEqual(len(ss), 1)
        self.assertEqual(ss[0].instruction, "walk forward")
        self.assertIsNone(ss[0].yaw_delta_target)
        self.assertIsNone(ss[0].proximity_target)

    def test_pipe_split(self):
        ss = parse_substeps("a | b | c", self.targets, self.t, self.thr)
        self.assertEqual([s.instruction for s in ss], ["a", "b", "c"])

    def test_then_split(self):
        ss = parse_substeps("a then b, then c", self.targets, self.t, self.thr)
        self.assertEqual([s.instruction for s in ss], ["a", "b", "c"])

    def test_semicolon_split(self):
        ss = parse_substeps("a; b; c", self.targets, self.t, self.thr)
        self.assertEqual(len(ss), 3)

    def test_mixed_separators(self):
        ss = parse_substeps("a | b, then c; d",
                             self.targets, self.t, self.thr)
        self.assertEqual([s.instruction for s in ss], ["a", "b", "c", "d"])

    def test_proximity_auto_detected_red(self):
        ss = parse_substeps("walk to the red box",
                             self.targets, self.t, self.thr)
        self.assertEqual(ss[0].proximity_target, (3.0, 0.0, 0.30))

    def test_proximity_auto_detected_blue(self):
        ss = parse_substeps("navigate to the blue box",
                             self.targets, self.t, self.thr)
        self.assertEqual(ss[0].proximity_target, (2.0, -1.5, 0.25))

    def test_proximity_picks_last_mention(self):
        # "past the blue box to the red box" → red
        ss = parse_substeps("walk past the blue box to the red box",
                             self.targets, self.t, self.thr)
        self.assertEqual(ss[0].proximity_target, (3.0, 0.0, 0.30))

    def test_yaw_target_left_deg(self):
        ss = parse_substeps("turn left 90 degrees",
                             self.targets, self.t, self.thr)
        self.assertAlmostEqual(ss[0].yaw_delta_target, math.pi / 2, places=5)
        self.assertIsNone(ss[0].proximity_target)

    def test_yaw_target_right_deg(self):
        ss = parse_substeps("turn right 45 deg",
                             self.targets, self.t, self.thr)
        self.assertAlmostEqual(ss[0].yaw_delta_target,
                                -math.radians(45.0), places=5)

    def test_yaw_target_radians(self):
        ss = parse_substeps("turn left 0.5 rad",
                             self.targets, self.t, self.thr)
        self.assertAlmostEqual(ss[0].yaw_delta_target, 0.5, places=5)

    def test_turn_with_target_prefers_yaw(self):
        # If both yaw and proximity are detected, planner prefers yaw —
        # turns are explicit primitives.
        ss = parse_substeps("turn right 90 deg toward the red box",
                             self.targets, self.t, self.thr)
        self.assertIsNotNone(ss[0].yaw_delta_target)
        self.assertIsNone(ss[0].proximity_target)

    def test_full_user_instruction(self):
        ss = parse_substeps(
            "walk forward until reaching the red box, then turn right 90 deg, "
            "then walk forward",
            self.targets, self.t, self.thr,
        )
        self.assertEqual(len(ss), 3)
        self.assertEqual(ss[0].proximity_target, (3.0, 0.0, 0.30))
        self.assertAlmostEqual(ss[1].yaw_delta_target, -math.pi / 2, places=5)
        self.assertIsNone(ss[2].proximity_target)
        self.assertIsNone(ss[2].yaw_delta_target)


# ===========================================================================
# Termination logic
# ===========================================================================


def _make_state(yaw0: float = 0.0) -> TerminationState:
    return TerminationState(
        step_idx=0, started_at=0.0, start_yaw=yaw0, last_yaw=yaw0,
    )


class CheckTerminationTests(unittest.TestCase):

    def test_yaw_target_reached_left(self):
        ss = SubStep("turn left", time_limit=10.0,
                     yaw_delta_target=math.radians(90.0))
        state = _make_state()
        state.yaw_unwrap = math.radians(86.0)  # within 5° tolerance
        reason = check_termination(
            ss, state, current_pos_xy=None, vlm_stop=False, now=1.0,
            yaw_tolerance_deg=5.0,
        )
        self.assertIsNotNone(reason)
        self.assertIn("yaw target reached", reason)

    def test_yaw_target_not_yet(self):
        ss = SubStep("turn left", time_limit=10.0,
                     yaw_delta_target=math.radians(90.0))
        state = _make_state()
        state.yaw_unwrap = math.radians(60.0)
        reason = check_termination(
            ss, state, current_pos_xy=None, vlm_stop=False, now=1.0,
        )
        self.assertIsNone(reason)

    def test_yaw_target_right(self):
        ss = SubStep("turn right", time_limit=10.0,
                     yaw_delta_target=-math.radians(90.0))
        state = _make_state()
        state.yaw_unwrap = -math.radians(85.0)  # within tolerance
        reason = check_termination(
            ss, state, current_pos_xy=None, vlm_stop=False, now=1.0,
            yaw_tolerance_deg=5.0,
        )
        self.assertIn("yaw target reached", reason or "")

    def test_proximity_reached(self):
        ss = SubStep("walk to red", proximity_target=(3.0, 0.0, 0.3),
                     proximity_threshold=1.0, time_limit=30.0)
        state = _make_state()
        reason = check_termination(
            ss, state, current_pos_xy=(2.5, 0.0),  # dist = 0.5
            vlm_stop=False, now=1.0,
        )
        self.assertIn("reached target", reason or "")

    def test_proximity_skipped_when_no_pose(self):
        ss = SubStep("walk to red", proximity_target=(3.0, 0.0, 0.3),
                     proximity_threshold=1.0, time_limit=30.0)
        state = _make_state()
        reason = check_termination(
            ss, state, current_pos_xy=None,  # ← real-robot, no odom
            vlm_stop=False, now=1.0,
        )
        self.assertIsNone(reason)

    def test_closest_approach_termination(self):
        ss = SubStep("walk to red", proximity_target=(3.0, 0.0, 0.3),
                     proximity_threshold=0.3, time_limit=30.0)
        state = _make_state()
        # Walk in: 2.5 → 1.0 → 0.5 (min) → 0.7 → 0.9 (retreating)
        for d in [2.5, 1.0, 0.5, 0.7]:
            r = check_termination(ss, state,
                                    current_pos_xy=(3.0 - d, 0.0),
                                    vlm_stop=False, now=1.0,
                                    closest_approach_min=1.5,
                                    closest_approach_margin=0.25)
            self.assertIsNone(r, f"unexpectedly done at d={d}: {r}")
        # Now 0.9 retreating — min was 0.5, +0.25 margin → fires at d>0.75
        r = check_termination(ss, state, current_pos_xy=(3.0 - 0.9, 0.0),
                              vlm_stop=False, now=1.0,
                              closest_approach_min=1.5,
                              closest_approach_margin=0.25)
        self.assertIn("closest approach", r or "")

    def test_vlm_stop(self):
        ss = SubStep("anything", time_limit=30.0)
        state = _make_state()
        reason = check_termination(
            ss, state, current_pos_xy=None, vlm_stop=True, now=1.0,
        )
        self.assertEqual(reason, "NaVILA stop")

    def test_time_limit(self):
        ss = SubStep("anything", time_limit=10.0)
        state = _make_state()
        reason = check_termination(
            ss, state, current_pos_xy=None, vlm_stop=False, now=10.5,
        )
        self.assertIn("time limit", reason or "")

    def test_priority_yaw_beats_others(self):
        ss = SubStep("turn", time_limit=1.0,                       # would fire on time
                     yaw_delta_target=math.radians(90.0))
        state = _make_state()
        state.yaw_unwrap = math.radians(95.0)                      # also fires on yaw
        reason = check_termination(
            ss, state, current_pos_xy=None, vlm_stop=True, now=10.0,
        )
        self.assertIn("yaw target reached", reason or "")          # yaw wins


# ===========================================================================
# Yaw accumulation
# ===========================================================================


class YawAccumulationTests(unittest.TestCase):

    def test_simple_accumulation(self):
        state = _make_state(yaw0=0.0)
        for y in [0.1, 0.2, 0.3, 0.4]:
            update_yaw_unwrap(state, y)
        self.assertAlmostEqual(state.yaw_unwrap, 0.4, places=5)

    def test_negative_accumulation(self):
        state = _make_state(yaw0=0.0)
        for y in [-0.1, -0.2, -0.3]:
            update_yaw_unwrap(state, y)
        self.assertAlmostEqual(state.yaw_unwrap, -0.3, places=5)

    def test_wraps_correctly_through_pi(self):
        # Simulate yaw going from +π−ε to −π+ε (a wrap from +pi to -pi).
        state = _make_state(yaw0=math.pi - 0.05)
        update_yaw_unwrap(state, -math.pi + 0.05)  # +0.1 rad change, not -2π+0.1
        self.assertAlmostEqual(state.yaw_unwrap, 0.1, places=4)

    def test_two_full_rotations(self):
        # 720° rotation should accumulate to ~4π.
        state = _make_state(yaw0=0.0)
        steps = 200
        for i in range(steps):
            y = wrap_pi((i + 1) * 4 * math.pi / steps)
            update_yaw_unwrap(state, y)
        self.assertAlmostEqual(state.yaw_unwrap, 4 * math.pi, places=2)


# ===========================================================================
# Controller dispatch
# ===========================================================================


class ApplyControllersTests(unittest.TestCase):

    def test_pure_turn_bypasses_vlm(self):
        ss = SubStep("turn", time_limit=10.0,
                     yaw_delta_target=-math.radians(90.0))
        state = _make_state()
        out = apply_controllers(
            ss=ss, state=state,
            current_pos_xy=None, current_yaw=0.0,
            vlm_cmd=(0.4, 0.0, 0.0),  # VLM says forward
            vx_max=0.6, vy_max=0.3, vyaw_max=0.6,
        )
        self.assertEqual(out.tag, "TURN")
        self.assertEqual(out.vx, 0.0)
        self.assertLess(out.vyaw, 0.0)               # turning right

    def test_turn_uses_min_floor(self):
        ss = SubStep("turn", time_limit=10.0,
                     yaw_delta_target=-math.radians(90.0))
        state = _make_state()
        # remaining ≈ -0.05 rad, K_p=2.0 → −0.1 rad/s. Floor of 0.30 must apply.
        state.yaw_unwrap = -math.radians(87.0)
        out = apply_controllers(
            ss=ss, state=state,
            current_pos_xy=None, current_yaw=0.0,
            vlm_cmd=(0.0, 0.0, 0.0),
            vx_max=0.6, vy_max=0.3, vyaw_max=0.6,
            turn_min_vyaw=0.30,
        )
        self.assertAlmostEqual(abs(out.vyaw), 0.30, places=2)

    def test_proximity_substep_uses_heading_assist(self):
        # K1 at origin facing +x, target at (3, 1.5) — bearing ≈ +27°
        # so heading-assist should emit a positive vyaw (turn left).
        ss = SubStep("walk to red", proximity_target=(3.0, 1.5, 0.3),
                     proximity_threshold=1.0, time_limit=30.0)
        state = _make_state()
        out = apply_controllers(
            ss=ss, state=state,
            current_pos_xy=(0.0, 0.0), current_yaw=0.0,
            vlm_cmd=(0.4, 0.0, 0.0),
            vx_max=0.6, vy_max=0.3, vyaw_max=0.6,
            heading_assist=True,
        )
        self.assertEqual(out.tag, "HEAD")
        self.assertGreater(out.vyaw, 0.0)            # turn left toward target

    def test_proximity_substep_zero_assist_when_aligned(self):
        # K1 already pointing right at the target → assist should be ~0.
        ss = SubStep("walk to red", proximity_target=(3.0, 0.0, 0.3))
        state = _make_state()
        out = apply_controllers(
            ss=ss, state=state,
            current_pos_xy=(0.0, 0.0), current_yaw=0.0,
            vlm_cmd=(0.4, 0.0, 0.0),
            vx_max=0.6, vy_max=0.3, vyaw_max=0.6,
            heading_assist=True,
        )
        self.assertEqual(out.tag, "HEAD")
        self.assertAlmostEqual(out.vyaw, 0.0, places=5)
        self.assertAlmostEqual(out.vx, 0.4, places=5)

    def test_no_pose_falls_through_to_vlm(self):
        ss = SubStep("walk to red", proximity_target=(3.0, 0.0, 0.3))
        state = _make_state()
        out = apply_controllers(
            ss=ss, state=state,
            current_pos_xy=None, current_yaw=None,   # ← no pose
            vlm_cmd=(0.4, 0.0, 0.1),
            vx_max=0.6, vy_max=0.3, vyaw_max=0.6,
            heading_assist=True,
        )
        self.assertEqual(out.tag, "VLM ")

    def test_vlm_clipping(self):
        ss = SubStep("walk forward")  # no proximity, no yaw → VLM passthrough
        state = _make_state()
        out = apply_controllers(
            ss=ss, state=state,
            current_pos_xy=None, current_yaw=None,
            vlm_cmd=(5.0, 5.0, 5.0),                 # absurdly large
            vx_max=0.4, vy_max=0.15, vyaw_max=0.4,
        )
        self.assertLessEqual(out.vx, 0.4)
        self.assertLessEqual(out.vy, 0.15)
        self.assertLessEqual(out.vyaw, 0.4)


# ===========================================================================
# Pure helpers
# ===========================================================================


class HelperTests(unittest.TestCase):

    def test_wrap_pi_basic(self):
        self.assertAlmostEqual(wrap_pi(0.0), 0.0)
        self.assertAlmostEqual(wrap_pi(math.pi), -math.pi, places=5)
        self.assertAlmostEqual(wrap_pi(-math.pi - 0.1), math.pi - 0.1, places=5)

    def test_wrap_pi_idempotent(self):
        for x in [0.5, 2.0, -3.0, 7.5]:
            self.assertAlmostEqual(wrap_pi(wrap_pi(x)), wrap_pi(x), places=10)

    def test_yaw_from_quat_identity(self):
        # w=1, x=y=z=0 → yaw=0
        self.assertAlmostEqual(yaw_from_quat([1.0, 0.0, 0.0, 0.0]), 0.0)

    def test_yaw_from_quat_pi_over_4(self):
        # 45° z-rotation: w=cos(π/8), z=sin(π/8)
        c, s = math.cos(math.pi / 8), math.sin(math.pi / 8)
        self.assertAlmostEqual(yaw_from_quat([c, 0.0, 0.0, s]),
                                math.pi / 4, places=5)


if __name__ == "__main__":
    unittest.main()
