"""NaVILA paper conformance tests.

Bug 1 (velocity mapping, §II-B) and Bug 2 (frame history sampling, §II-A)
asserted directly against the paper's prescriptions. These are the tests
the user requested in the bug-fix brief; they're self-contained and have
no torch / llava / SDK dependencies so they can run on any laptop.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navila_k1_bridge import (  # noqa: E402
    FORWARD_SPEED, TURN_SPEED, build_prompt, parse_action,
)
from navila_k1_core import NUM_FRAMES, VLMRunner  # noqa: E402


# ===========================================================================
# BUG 1 — velocity mapping (NaVILA paper §II-B)
# ===========================================================================
#
# Paper:
#   "VLM outputs a fixed set of actionable words ... we cast these
#    instructions to fixed command velocities {0.5 m/s, π/6 rad/s,
#    −π/6 rad/s, 0} and execute with corresponding time durations to
#    align with the specific VLM value."
#
# i.e. speed is FIXED, duration scales with distance / angle.
# ===========================================================================


class Bug1VelocityMappingTests(unittest.TestCase):

    # --- fixed-speed assertion: speed is ALWAYS 0.5 m/s / π/6 rad/s --------

    def test_forward_speed_is_fixed_at_half_mps(self):
        for cm in (10, 25, 50, 75, 100, 200, 500):
            vx, _, _, _, _ = parse_action(f"move forward {cm} cm")
            self.assertAlmostEqual(vx, 0.5, places=6,
                                    msg=f"forward {cm}cm should give vx=0.5")

    def test_turn_speed_is_fixed_at_pi_over_six_rad_per_sec(self):
        for deg in (10, 30, 45, 60, 90, 180):
            _, _, vyaw_left, _, _ = parse_action(f"turn left {deg} degrees")
            _, _, vyaw_right, _, _ = parse_action(f"turn right {deg} degrees")
            self.assertAlmostEqual(vyaw_left, math.pi / 6, places=6,
                                    msg=f"turn left {deg}° should give "
                                         "vyaw=+π/6")
            self.assertAlmostEqual(vyaw_right, -math.pi / 6, places=6,
                                    msg=f"turn right {deg}° should give "
                                         "vyaw=-π/6")

    # --- duration = distance / speed ---------------------------------------

    def test_forward_75cm_duration_is_1p5s(self):
        vx, _, _, dur, _ = parse_action("move forward 75 cm")
        self.assertAlmostEqual(vx, 0.5, places=6)
        self.assertAlmostEqual(dur, 1.5, places=6)

    def test_forward_50cm_duration_is_1s(self):
        vx, _, _, dur, _ = parse_action("move forward 50 cm")
        self.assertAlmostEqual(vx, 0.5, places=6)
        self.assertAlmostEqual(dur, 1.0, places=6)

    def test_forward_25cm_duration_is_0p5s(self):
        vx, _, _, dur, _ = parse_action("move forward 25 cm")
        self.assertAlmostEqual(vx, 0.5, places=6)
        self.assertAlmostEqual(dur, 0.5, places=6)

    def test_turn_left_30deg_duration_is_1s(self):
        # π/6 rad / (π/6 rad/s) = 1 s.
        _, _, vyaw, dur, _ = parse_action("turn left 30 degrees")
        self.assertAlmostEqual(vyaw, math.pi / 6, places=6)
        self.assertAlmostEqual(dur, 1.0, places=6)

    def test_turn_left_60deg_duration_is_2s(self):
        _, _, vyaw, dur, _ = parse_action("turn left 60 degrees")
        self.assertAlmostEqual(vyaw, math.pi / 6, places=6)
        self.assertAlmostEqual(dur, 2.0, places=6)

    def test_turn_right_30deg_duration_is_1s(self):
        _, _, vyaw, dur, _ = parse_action("turn right 30 degrees")
        self.assertAlmostEqual(vyaw, -math.pi / 6, places=6)
        self.assertAlmostEqual(dur, 1.0, places=6)

    def test_turn_right_60deg_duration_is_2s(self):
        _, _, vyaw, dur, _ = parse_action("turn right 60 degrees")
        self.assertAlmostEqual(vyaw, -math.pi / 6, places=6)
        self.assertAlmostEqual(dur, 2.0, places=6)

    # --- stop produces zero velocity ---------------------------------------

    def test_stop_is_zero_velocity_zero_duration(self):
        vx, vy, vyaw, dur, label = parse_action("stop")
        self.assertEqual((vx, vy, vyaw), (0.0, 0.0, 0.0))
        self.assertEqual(dur, 0.0)
        self.assertEqual(label, "stop")

    # --- distance covered = speed × duration (sanity) ----------------------

    def test_distance_covered_matches_request(self):
        # Open-loop, no clipping by VX_MAX (paper-spec speed equals our
        # FORWARD_SPEED): distance = vx × dur.
        for cm in (25, 50, 75, 100, 150):
            vx, _, _, dur, _ = parse_action(f"move forward {cm} cm")
            self.assertAlmostEqual(abs(vx) * dur, cm / 100.0, places=6,
                                    msg=f"forward {cm}cm should cover "
                                         f"{cm/100:.2f} m, got "
                                         f"{abs(vx) * dur:.4f}")

    def test_angle_covered_matches_request(self):
        for deg in (15, 30, 45, 60, 90, 180):
            _, _, vyaw, dur, _ = parse_action(f"turn left {deg} degrees")
            self.assertAlmostEqual(abs(vyaw) * dur, math.radians(deg),
                                    places=6,
                                    msg=f"turn left {deg}° should cover "
                                         f"{math.radians(deg):.4f} rad, got "
                                         f"{abs(vyaw) * dur:.4f}")


# ===========================================================================
# BUG 2 — frame history sampling (NaVILA paper §II-A)
# ===========================================================================
#
# Paper:
#   "We first extract the most recent frame t as the current observation
#    and then uniformly sample frames from the preceding t−1 frames,
#    ensuring the first frame is always included."
#
# Reference impl: llava/mm_utils.get_frame_from_vcap_vlnce uses
#   np.linspace(0, len-1, num=N-1, endpoint=False, dtype=int)
# which always includes index 0 (the first frame).
# ===========================================================================


class Bug2FrameHistoryTests(unittest.TestCase):

    @staticmethod
    def _make_frames(n: int) -> list:
        """Make n distinct sentinel objects. They don't need to be PIL
        images for sampling logic — VLMRunner.sample_frames treats them
        opaquely."""
        return [f"frame_{i:02d}" for i in range(n)]

    def test_first_frame_always_included_at_index_0(self):
        # Paper requirement: ensure the first frame is always part of the
        # historical context, regardless of how long the episode has run.
        for total in (8, 12, 20, 50, 200):
            frames = self._make_frames(total)
            sampled = VLMRunner.sample_frames(frames, num_frames=NUM_FRAMES)
            self.assertEqual(len(sampled), NUM_FRAMES,
                              f"len(frames)={total}: expected 8 sampled")
            self.assertEqual(sampled[0], "frame_00",
                              f"len(frames)={total}: first frame missing")

    def test_current_frame_is_last_in_buffer(self):
        for total in (8, 12, 20, 50, 200):
            frames = self._make_frames(total)
            sampled = VLMRunner.sample_frames(frames, num_frames=NUM_FRAMES)
            self.assertEqual(sampled[-1], f"frame_{total - 1:02d}",
                              f"len(frames)={total}: current is "
                              f"not the latest frame")

    def test_exactly_7_historical_plus_1_current(self):
        frames = self._make_frames(20)
        sampled = VLMRunner.sample_frames(frames, num_frames=NUM_FRAMES)
        self.assertEqual(len(sampled), 8)
        # The first 7 are "historical observations", the 8th is "current".
        historical, current = sampled[:7], sampled[7]
        self.assertEqual(len(historical), 7)
        self.assertEqual(current, frames[-1])

    def test_current_frame_separate_from_historical(self):
        # With more than NUM_FRAMES frames in the buffer, the current
        # frame should not appear in the historical set.
        frames = self._make_frames(20)
        sampled = VLMRunner.sample_frames(frames, num_frames=NUM_FRAMES)
        historical, current = sampled[:7], sampled[7]
        self.assertNotIn(current, historical,
                          "current frame must not appear in historical "
                          "(paper §II-A: 'current and historical "
                          "observations serve different roles')")

    def test_historical_frames_are_uniformly_spaced_not_just_last_7(self):
        # If the implementation just took the last 7 frames, the sampled
        # historical set would be {frame_13...frame_19}. The paper says
        # uniform sampling, so we should see frames spread across the
        # whole episode.
        frames = self._make_frames(20)
        sampled = VLMRunner.sample_frames(frames, num_frames=NUM_FRAMES)
        historical = sampled[:7]
        # If the impl is paper-correct, frame_00 is in there AND the
        # historical frames are NOT {13,14,15,16,17,18,19}.
        last_seven = set(f"frame_{i:02d}" for i in range(13, 20))
        self.assertNotEqual(set(historical), last_seven,
                             "historical = last 7 frames is the bug we "
                             "are guarding against")
        self.assertIn("frame_00", historical)
        # Sanity: indices in the historical set should span more than
        # half the episode.
        idxs = sorted(int(f.split("_")[1]) for f in historical)
        self.assertGreaterEqual(max(idxs) - min(idxs), 10,
                                 f"historical span too small: {idxs}")

    def test_matches_reference_linspace_exactly(self):
        # The exact indices the reference NaVILA repo's
        # `get_frame_from_vcap_vlnce` produces.
        import numpy as np
        for total in (8, 9, 10, 15, 20, 37, 200):
            frames = self._make_frames(total)
            sampled = VLMRunner.sample_frames(frames, num_frames=NUM_FRAMES)
            expected_indices = list(
                np.linspace(0, total - 1, num=NUM_FRAMES - 1,
                            endpoint=False, dtype=int)
            ) + [total - 1]
            actual_indices = [int(f.split("_")[1]) for f in sampled]
            self.assertEqual(actual_indices, expected_indices,
                              f"len={total}: indices {actual_indices} "
                              f"!= reference {expected_indices}")

    def test_short_episode_pads_with_first_frame(self):
        # Before we've accumulated NUM_FRAMES frames, pad on the left with
        # the first frame (so the model sees a stable bootstrap window).
        frames = self._make_frames(3)
        sampled = VLMRunner.sample_frames(frames, num_frames=NUM_FRAMES)
        self.assertEqual(len(sampled), NUM_FRAMES)
        self.assertEqual(sampled[-3:],
                          ["frame_00", "frame_01", "frame_02"])
        self.assertTrue(all(f == "frame_00" for f in sampled[:5]),
                         f"left-pad should be the first frame, got {sampled[:5]}")

    def test_no_dedup_no_two_currents(self):
        # With a long episode the same frame must not appear as both a
        # historical and the current observation (the paper distinguishes
        # them, and showing the same frame twice wastes a slot).
        frames = self._make_frames(50)
        sampled = VLMRunner.sample_frames(frames, num_frames=NUM_FRAMES)
        historical, current = sampled[:7], sampled[7]
        self.assertNotIn(current, historical)


# ===========================================================================
# BUG 2 — prompt textual cues
# ===========================================================================


class Bug2PromptCueTests(unittest.TestCase):

    def test_prompt_contains_required_textual_cues(self):
        prompt = build_prompt("walk to the red box", num_frames=NUM_FRAMES)
        self.assertIn("historical observations", prompt,
                       "paper §II-A mandates a 'historical observations' cue")
        self.assertIn("current observation", prompt,
                       "paper §II-A mandates a 'current observation' cue")

    def test_prompt_historical_appears_before_current(self):
        prompt = build_prompt("walk to the red box", num_frames=NUM_FRAMES)
        hist_idx = prompt.index("historical observations")
        curr_idx = prompt.index("current observation")
        self.assertLess(hist_idx, curr_idx,
                         "historical context must come before the current "
                         "observation token in the prompt")

    def test_prompt_includes_instruction(self):
        prompt = build_prompt("walk to the red box", num_frames=NUM_FRAMES)
        self.assertIn("walk to the red box", prompt)

    def test_prompt_matches_reference_run_navigation(self):
        # Byte-for-byte check against booster/NaVILA/llava/eval/run_navigation.py
        # (the inference script the trained model expects).
        ref_path = (Path.home() / "Projects/k1_research/booster/NaVILA"
                    "/llava/eval/run_navigation.py")
        if not ref_path.exists():
            self.skipTest(f"reference inference script not at {ref_path}")
        ref_text = ref_path.read_text()
        for needle in (
            "Imagine you are a robot programmed for navigation tasks.",
            "of historical observations",
            "and current observation",
            "Your assigned task is",
            "Analyze this series of images",
        ):
            self.assertIn(needle, ref_text,
                          f"reference does not contain {needle!r}")
        ours = build_prompt("X", num_frames=NUM_FRAMES)
        for needle in (
            "Imagine you are a robot programmed for navigation tasks.",
            "of historical observations",
            "and current observation",
            "Your assigned task is",
            "Analyze this series of images",
        ):
            self.assertIn(needle, ours,
                          f"build_prompt does not contain {needle!r}")


# ===========================================================================
# Integration smoke — end-to-end frame buffer + sampling through VLMRunner
# ===========================================================================


class FrameBufferIntegrationTests(unittest.TestCase):
    """Drive a fake VLMRunner with frames, verify the inference fn sees
    paper-correct frame samples without spinning up torch / llava."""

    def setUp(self):
        # No model load — we inject a fake inference fn.
        self.runner = VLMRunner(model_path=Path("/dev/null"))

    def test_buffer_unbounded_first_frame_preserved(self):
        from PIL import Image
        self.runner.bootstrap_buffer(Image.new("RGB", (4, 4), (255, 0, 0)))
        for i in range(50):
            color = (0, (i * 5) % 255, 0)
            self.runner.push_frame(Image.new("RGB", (4, 4), color))
        snap = self.runner._snapshot()
        # Bootstrap pushes ONE frame now (not 8); plus 50 real pushes.
        self.assertEqual(len(snap), 51)
        # First frame must still be the red bootstrap.
        self.assertEqual(snap[0].getpixel((0, 0)), (255, 0, 0))

    def test_inference_receives_8_frames_with_first_preserved(self):
        from PIL import Image
        seen_frames: list = []

        def fake_inference(frames, instruction):
            seen_frames.append(list(frames))
            return "stop"  # ends the action immediately

        self.runner.set_inference_fn(fake_inference)
        # Seed the buffer (no model loaded — inject fake inference fn).
        red = Image.new("RGB", (4, 4), (255, 0, 0))
        self.runner.bootstrap_buffer(red)
        for i in range(20):
            color = (i * 10, 0, 0)
            self.runner.push_frame(Image.new("RGB", (4, 4), color))
        self.runner.set_instruction("test")
        self.runner.start()
        # Wait for at least one inference cycle.
        import time
        deadline = time.perf_counter() + 2.0
        while not seen_frames and time.perf_counter() < deadline:
            time.sleep(0.02)
        self.runner.shutdown(timeout=1.0)

        self.assertTrue(seen_frames, "fake inference was never called")
        first_call = seen_frames[0]
        self.assertEqual(len(first_call), NUM_FRAMES)
        # Red bootstrap must be at index 0.
        self.assertEqual(first_call[0].getpixel((0, 0)), (255, 0, 0))


# ===========================================================================
# Hold-for-duration / settle-between-actions integration
# ===========================================================================


class VLMRunnerHoldCycleTests(unittest.TestCase):
    """Verifies the paper's hold-then-zero loop: VLMRunner publishes the
    fixed velocity, holds it for ~duration seconds, publishes zeros, then
    asks the VLM again — all without torch / llava."""

    def test_publish_then_zero_for_forward_25cm(self):
        # "move forward 25 cm" → publishes (0.5, 0, 0), holds it for
        # ~0.5s, then publishes zeros before the next inference.
        import threading
        import time
        from PIL import Image
        from navila_k1_core import VLMRunner

        runner = VLMRunner(model_path=Path("/dev/null"),
                            vx_max=1.0, vy_max=1.0, vyaw_max=1.0)
        calls = [0]
        inf_done = threading.Event()
        loop_done = threading.Event()

        def fake_inference(frames, instruction):
            calls[0] += 1
            if calls[0] == 1:
                inf_done.set()
                return "move forward 25 cm"  # 0.5 s hold
            loop_done.set()
            # Block the second inference so we can capture the post-hold
            # zero-cmd state without it being overwritten immediately.
            time.sleep(0.5)
            return "stop"

        runner.set_inference_fn(fake_inference)
        runner.bootstrap_buffer(Image.new("RGB", (4, 4), (0, 0, 0)))
        runner.set_instruction("test")
        runner.start()

        # Wait for the first inference to return + publish to land.
        self.assertTrue(inf_done.wait(timeout=2.0))
        time.sleep(0.05)
        with runner._lock:
            during_hold = runner._cmd
        # Wait for the hold to expire and the second inference to start
        # (which means the zero-publish already happened just before).
        self.assertTrue(loop_done.wait(timeout=2.0))
        time.sleep(0.02)
        with runner._lock:
            after_hold = runner._cmd
        runner.shutdown(timeout=2.0)

        self.assertAlmostEqual(during_hold[0], 0.5, places=3,
                                msg=f"during hold vx={during_hold[0]} "
                                     "expected 0.5")
        self.assertEqual(during_hold[1], 0.0)
        self.assertEqual(during_hold[2], 0.0)
        self.assertEqual(after_hold, (0.0, 0.0, 0.0),
                          "expected zeros published between actions, "
                          f"got {after_hold}")

    def test_hold_duration_scales_with_distance(self):
        # Holding "move forward 75 cm" should take ~1.5s; "move forward
        # 25 cm" should take ~0.5s. We measure the wall-clock interval
        # between two consecutive inference calls.
        import time
        from PIL import Image
        from navila_k1_core import VLMRunner

        for cm, expected_hold in [(25, 0.5), (75, 1.5)]:
            runner = VLMRunner(model_path=Path("/dev/null"),
                                vx_max=1.0, vy_max=1.0, vyaw_max=1.0)
            t_starts: list = []
            calls = [0]

            def fake_inference(frames, instruction, _cm=cm):
                calls[0] += 1
                t_starts.append(time.perf_counter())
                if calls[0] == 1:
                    return f"move forward {_cm} cm"
                return "stop"

            runner.set_inference_fn(fake_inference)
            runner.bootstrap_buffer(Image.new("RGB", (4, 4)))
            runner.set_instruction("test")
            runner.start()
            deadline = time.perf_counter() + expected_hold + 2.0
            while calls[0] < 2 and time.perf_counter() < deadline:
                time.sleep(0.02)
            runner.shutdown(timeout=2.0)

            self.assertGreaterEqual(calls[0], 2)
            interval = t_starts[1] - t_starts[0]
            # Interval ≈ hold + settle (0.1s). Tolerate ±25%.
            self.assertGreater(interval, expected_hold * 0.9,
                                f"hold for {cm}cm too short: "
                                f"{interval:.3f}s vs expected≥{expected_hold:.2f}s")
            self.assertLess(interval, expected_hold + 0.8,
                             f"hold for {cm}cm too long: "
                             f"{interval:.3f}s vs expected~{expected_hold:.2f}s")


# ===========================================================================
# Audit fixes — duration stretch on clip, sleep-after-stop, buffer compaction
# ===========================================================================


class DurationStretchOnClipTests(unittest.TestCase):
    """When the operator's safety cap clips a paper-spec command speed,
    the hold duration must stretch so the requested distance / angle is
    still achieved."""

    def test_vyaw_clipped_to_safety_cap_stretches_duration(self):
        # Real-robot default vyaw_max=0.4 < π/6 ≈ 0.524. A 30° turn at
        # paper-spec (0.524 rad/s for 1.0 s) becomes 0.4 rad/s for
        # ~1.31 s under the cap. Angle stays at 30° regardless.
        import math, threading, time
        from PIL import Image
        from navila_k1_core import VLMRunner

        observed: dict = {}
        cv = threading.Event()

        def fake_inference(frames, instruction):
            return "turn left 30 degrees" if not cv.is_set() else "stop"

        runner = VLMRunner(model_path=Path("/dev/null"),
                            vx_max=0.4, vy_max=0.15, vyaw_max=0.4)
        runner.set_inference_fn(fake_inference)
        runner.bootstrap_buffer(Image.new("RGB", (4, 4)))
        runner.set_instruction("test")

        # We can't easily intercept the loop's internal `duration` value,
        # so instead: time the gap between first publish and zero-publish
        # by polling the runner's _cmd. (The settle adds ~0.1 s.)
        publishes: list = []

        def watcher():
            last = runner.get_command()
            t0 = time.perf_counter()
            while not cv.is_set():
                cmd = runner.get_command()
                if cmd != last:
                    publishes.append((time.perf_counter() - t0, cmd))
                    last = cmd
                time.sleep(0.005)

        wt = threading.Thread(target=watcher, daemon=True)
        wt.start()
        runner.start()
        time.sleep(2.5)
        cv.set()
        wt.join(timeout=1.0)
        runner.shutdown(timeout=2.0)

        # Find the forward publish and the subsequent zero publish.
        forward_t = next((t for t, c in publishes
                          if abs(c[2] - 0.4) < 1e-3), None)
        zero_t = next((t for t, c in publishes
                       if forward_t is not None and t > forward_t
                       and c == (0.0, 0.0, 0.0)), None)
        self.assertIsNotNone(forward_t, f"no clipped vyaw publish in {publishes}")
        self.assertIsNotNone(zero_t,
                              f"no zero publish after forward in {publishes}")
        hold = zero_t - forward_t
        # Paper-spec duration is π/6 / (π/6) = 1.0 s; stretched by
        # 0.524/0.4 ≈ 1.31 → ~1.31 s. Allow ±0.15 s slop for scheduling.
        expected = math.radians(30) / 0.4  # = 1.309
        self.assertGreater(hold, expected - 0.2,
                            f"hold {hold:.3f}s shorter than stretched "
                            f"target {expected:.3f}s — clip not compensated")
        self.assertLess(hold, expected + 0.3,
                         f"hold {hold:.3f}s much longer than target "
                         f"{expected:.3f}s")

    def test_no_stretch_when_no_clipping(self):
        # When caps are above paper-spec, duration stays equal to
        # distance/speed (no stretch).
        import math, threading, time
        from PIL import Image
        from navila_k1_core import VLMRunner

        cv = threading.Event()

        def fake_inference(frames, instruction):
            return "turn left 30 degrees" if not cv.is_set() else "stop"

        runner = VLMRunner(model_path=Path("/dev/null"),
                            vx_max=1.0, vy_max=1.0, vyaw_max=1.0)
        runner.set_inference_fn(fake_inference)
        runner.bootstrap_buffer(Image.new("RGB", (4, 4)))
        runner.set_instruction("test")

        publishes: list = []
        last = runner.get_command()

        def watcher():
            nonlocal last
            t0 = time.perf_counter()
            while not cv.is_set():
                cmd = runner.get_command()
                if cmd != last:
                    publishes.append((time.perf_counter() - t0, cmd))
                    last = cmd
                time.sleep(0.005)

        import threading as _t
        wt = _t.Thread(target=watcher, daemon=True)
        wt.start()
        runner.start()
        time.sleep(2.0)
        cv.set()
        wt.join(timeout=1.0)
        runner.shutdown(timeout=2.0)

        forward_t = next((t for t, c in publishes
                          if abs(c[2] - math.pi / 6) < 1e-3), None)
        zero_t = next((t for t, c in publishes
                       if forward_t is not None and t > forward_t
                       and c == (0.0, 0.0, 0.0)), None)
        self.assertIsNotNone(forward_t, f"no paper-spec publish in {publishes}")
        self.assertIsNotNone(zero_t)
        hold = zero_t - forward_t
        # No clip → hold should be ~1.0 s (π/6 / (π/6)).
        self.assertGreater(hold, 0.8)
        self.assertLess(hold, 1.3)


class BufferCompactionTests(unittest.TestCase):

    def test_compact_preserves_first_and_recent(self):
        from navila_k1_core import VLMRunner
        buf = [f"f_{i}" for i in range(1000)]
        compacted = VLMRunner._compact_buffer(buf, cap=500, recent_keep=50)
        self.assertLessEqual(len(compacted), 500)
        self.assertEqual(compacted[0], "f_0", "first frame must survive")
        self.assertEqual(compacted[-50:],
                          [f"f_{i}" for i in range(950, 1000)],
                          "recent frames must survive intact")

    def test_compact_no_op_when_under_cap(self):
        from navila_k1_core import VLMRunner
        buf = [f"f_{i}" for i in range(100)]
        compacted = VLMRunner._compact_buffer(buf, cap=500, recent_keep=50)
        self.assertEqual(compacted, buf, "no compaction needed under cap")

    def test_compact_middle_is_uniformly_sampled(self):
        from navila_k1_core import VLMRunner
        buf = [f"f_{i}" for i in range(2000)]
        compacted = VLMRunner._compact_buffer(buf, cap=500, recent_keep=50)
        # 1 first + 449 middle + 50 recent = 500
        self.assertEqual(len(compacted), 500)
        middle = compacted[1:-50]
        # Middle should span roughly indices [1, 1949]
        idxs = [int(f.split("_")[1]) for f in middle]
        self.assertEqual(idxs[0], 1)
        # Last middle frame should be near the start of the "recent"
        # cutoff (= index 1949).
        self.assertLessEqual(abs(idxs[-1] - 1949), 5)
        # And the gaps between consecutive middle indices should be ~roughly
        # uniform (no big clumps).
        gaps = [b - a for a, b in zip(idxs[:-1], idxs[1:])]
        self.assertLessEqual(max(gaps) - min(gaps), 2)

    def test_push_triggers_compaction_at_cap(self):
        from PIL import Image
        from navila_k1_core import VLMRunner
        runner = VLMRunner(model_path=Path("/dev/null"))
        runner._buf_soft_cap = 50
        runner._buf_recent_keep = 10
        runner.bootstrap_buffer(Image.new("RGB", (4, 4), (255, 0, 0)))
        for _ in range(100):
            runner.push_frame(Image.new("RGB", (4, 4), (0, 0, 255)))
        self.assertLessEqual(len(runner._frame_buffer), 50)
        # The bootstrap red frame must survive.
        self.assertEqual(runner._frame_buffer[0].getpixel((0, 0)),
                          (255, 0, 0))

    def test_sample_frames_still_works_after_compaction(self):
        from PIL import Image
        from navila_k1_core import VLMRunner
        runner = VLMRunner(model_path=Path("/dev/null"))
        runner._buf_soft_cap = 50
        runner._buf_recent_keep = 10
        runner.bootstrap_buffer(Image.new("RGB", (4, 4), (255, 0, 0)))
        for _ in range(200):
            runner.push_frame(Image.new("RGB", (4, 4), (0, 255, 0)))
        sampled = runner._sample_frames_for_inference()
        self.assertEqual(len(sampled), NUM_FRAMES)
        self.assertEqual(sampled[0].getpixel((0, 0)), (255, 0, 0),
                          "first frame still at index 0 after compaction")


class StopSleepTests(unittest.TestCase):

    def test_stop_does_not_busy_spin(self):
        # When VLM emits stop, the loop should sleep briefly before
        # running the next inference.
        import time
        from PIL import Image
        from navila_k1_core import VLMRunner

        calls = [0]
        t_calls: list = []

        def fake_inference(frames, instruction):
            calls[0] += 1
            t_calls.append(time.perf_counter())
            return "stop"

        runner = VLMRunner(model_path=Path("/dev/null"))
        runner.set_inference_fn(fake_inference)
        runner.bootstrap_buffer(Image.new("RGB", (4, 4)))
        runner.set_instruction("test")
        runner.start()
        time.sleep(0.5)
        runner.shutdown(timeout=1.0)

        # In 0.5 s of pure stop, even with zero-cost fake inference, the
        # ~0.1 s sleep should cap us at well under 50 calls.
        self.assertLess(calls[0], 10,
                         f"{calls[0]} inferences in 0.5 s — busy-spin "
                         "after stop")
        # Verify consecutive-call gap is at least 80 ms.
        gaps = [b - a for a, b in zip(t_calls[:-1], t_calls[1:])]
        if gaps:
            self.assertGreater(min(gaps), 0.08,
                                f"min gap {min(gaps):.3f}s too small")


if __name__ == "__main__":
    unittest.main()
