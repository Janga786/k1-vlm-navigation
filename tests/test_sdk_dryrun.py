"""Test the actuator wrappers (Print/Dry/Live) with a mocked SDK.

The point: catch SDK-wrapper bugs (wrong mode transitions, missing
shutdown, watchdog not firing) without needing the real
``booster_robotics_sdk_python`` install.
"""

from __future__ import annotations

import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fake SDK (module-level; installed/restored in setUp/tearDown)
# ---------------------------------------------------------------------------


class _FakeRobotMode:
    kPrepare = "kPrepare"
    kWalking = "kWalking"
    kDamping = "kDamping"


class _FakeChannelFactory:
    _instance = None

    @classmethod
    def Instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.init_calls: list[tuple] = []

    def Init(self, *args):
        self.init_calls.append(args)


class _FakeGetModeResponse:
    def __init__(self):
        self.mode = None


class _FakeB1LocoClient:
    """Mimics the real binding: ChangeMode/Move/GetMode return an int rc,
    and the robot honors a mode switch only when it would in reality
    (configurable via changemode_rc / honor_changemode)."""

    def __init__(self):
        self.init_called = False
        self.move_calls: list[tuple] = []
        self.mode_calls: list = []
        self.current_mode = _FakeRobotMode.kPrepare  # standing, ready
        self.changemode_rc = 0        # nonzero = explicit refusal
        self.honor_changemode = True  # False = rc 0 but mode doesn't take

    def Init(self):
        self.init_called = True

    def ChangeMode(self, mode):
        self.mode_calls.append(mode)
        if self.changemode_rc == 0 and self.honor_changemode:
            self.current_mode = mode
        return self.changemode_rc

    def Move(self, vx, vy, vyaw):
        self.move_calls.append((vx, vy, vyaw))
        return 0

    def GetMode(self, resp):
        resp.mode = self.current_mode
        return 0


def _install_fake_sdk():
    """Plant a fake module so `from booster_robotics_sdk_python import ...` works."""
    _FakeChannelFactory._instance = None        # reset between tests
    fake = types.ModuleType("booster_robotics_sdk_python")
    fake.B1LocoClient = _FakeB1LocoClient
    fake.ChannelFactory = _FakeChannelFactory
    fake.RobotMode = _FakeRobotMode
    fake.GetModeResponse = _FakeGetModeResponse
    sys.modules["booster_robotics_sdk_python"] = fake
    return fake


def _uninstall_fake_sdk():
    sys.modules.pop("booster_robotics_sdk_python", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class PrintActuatorTests(unittest.TestCase):
    """The print actuator must NEVER touch the SDK."""

    def setUp(self):
        # Make sure even an accidental import would fail loudly:
        sys.modules.pop("booster_robotics_sdk_python", None)
        from navila_k1_realrobot import PrintActuator
        self.act = PrintActuator()

    def test_send_does_nothing_dangerous(self):
        self.act.init()                  # no-op
        self.act.send(0.5, 0.0, 0.0)     # just prints
        self.act.send(0.0, 0.0, 0.0)
        self.act.shutdown()              # no-op
        self.assertNotIn("booster_robotics_sdk_python", sys.modules)


class DryRunActuatorTests(unittest.TestCase):

    def setUp(self):
        self.fake_sdk = _install_fake_sdk()
        from navila_k1_realrobot import DryRunActuator
        self.act = DryRunActuator(net="eno1")

    def tearDown(self):
        _uninstall_fake_sdk()

    def test_init_creates_client_but_does_not_walk(self):
        self.act.init()
        client: _FakeB1LocoClient = self.act._client
        self.assertTrue(client.init_called)
        # The whole point of dry mode: NEVER ChangeMode to kWalking.
        self.assertEqual(client.mode_calls, [])

    def test_send_does_not_call_move(self):
        self.act.init()
        self.act.send(0.5, 0.0, 0.0)
        self.act.send(0.0, 0.0, 0.5)
        client: _FakeB1LocoClient = self.act._client
        self.assertEqual(client.move_calls, [],
                          "DryRunActuator must NEVER call Move()")

    def test_channel_factory_initialised_with_local_iface(self):
        # --net is the LOCAL interface NAME (FastDDS), never the robot IP.
        self.act.init()
        cf = _FakeChannelFactory.Instance()
        self.assertEqual(cf.init_calls, [(0, "eno1")])


class LiveActuatorTests(unittest.TestCase):

    def setUp(self):
        self.fake_sdk = _install_fake_sdk()
        from navila_k1_realrobot import LiveActuator
        # Use a fast send_hz so the test doesn't take long.
        self.act = LiveActuator(net="eno1",
                                 send_hz=100.0, watchdog_seconds=0.2)

    def tearDown(self):
        try:
            self.act.shutdown()
        except Exception:
            pass
        _uninstall_fake_sdk()

    def test_init_switches_to_walking(self):
        self.act.init()
        client: _FakeB1LocoClient = self.act._client
        self.assertTrue(client.init_called)
        self.assertEqual(client.mode_calls, [_FakeRobotMode.kWalking])

    def test_init_raises_when_changemode_refused(self):
        # A robot not in Prepare returns a nonzero rc — must fail LOUDLY at
        # launch, not stream Move() into the void ('robot did not move').
        from navila_k1_realrobot import LiveActuator
        act = LiveActuator(net="eno1", send_hz=100.0, watchdog_seconds=0.2)

        # Pre-build a client whose ChangeMode refuses.
        import navila_k1_realrobot as rr  # noqa: F401 (fake sdk already installed)
        with mock.patch.object(_FakeB1LocoClient, "ChangeMode",
                               autospec=True) as cm:
            def refuse(self_c, mode):
                self_c.mode_calls.append(mode)
                return 3
            cm.side_effect = refuse
            with self.assertRaises(SystemExit) as ctx:
                act.init()
            self.assertIn("Prepare", str(ctx.exception))

    def test_init_raises_when_mode_does_not_take(self):
        # rc == 0 but GetMode still reports kPrepare (silent refusal) — the
        # GetMode verification must catch it.
        from navila_k1_realrobot import LiveActuator
        act = LiveActuator(net="eno1", send_hz=100.0, watchdog_seconds=0.2)
        with mock.patch.object(_FakeB1LocoClient, "ChangeMode",
                               autospec=True) as cm:
            def silently_ignore(self_c, mode):
                self_c.mode_calls.append(mode)
                return 0  # claims success, but current_mode never changes
            cm.side_effect = silently_ignore
            with self.assertRaises(SystemExit) as ctx:
                act.init()
            self.assertIn("NOT", str(ctx.exception))

    def test_send_calls_move_at_send_hz(self):
        self.act.init()
        self.act.send(0.3, 0.0, 0.1)
        time.sleep(0.15)                                     # ≥ 10 ticks at 100 Hz
        client: _FakeB1LocoClient = self.act._client
        self.assertGreater(len(client.move_calls), 5,
                            f"Expected several Move() calls, got "
                            f"{len(client.move_calls)}")
        # The recently sent value should appear among the calls.
        self.assertIn((0.3, 0.0, 0.1), client.move_calls)

    def test_watchdog_zeros_stale_command(self):
        self.act.init()
        self.act.send(0.4, 0.0, 0.0)
        time.sleep(0.5)                                      # > watchdog (0.2s)
        client: _FakeB1LocoClient = self.act._client
        # The most recent Move() call should be (0,0,0) because the cmd is stale.
        self.assertEqual(client.move_calls[-1], (0.0, 0.0, 0.0))

    def test_shutdown_zeroes_and_stands(self):
        self.act.init()
        self.act.send(0.4, 0.0, 0.0)
        time.sleep(0.05)
        self.act.shutdown()
        client: _FakeB1LocoClient = self.act._client
        # Cleanup must have called Move(0,0,0) then ChangeMode(kPrepare) —
        # kPrepare keeps the biped STANDING; kDamping would make it collapse.
        self.assertEqual(client.move_calls[-1], (0.0, 0.0, 0.0))
        self.assertEqual(client.mode_calls,
                          [_FakeRobotMode.kWalking, _FakeRobotMode.kPrepare])

    def test_shutdown_idempotent_on_send_failure(self):
        self.act.init()
        # Make Move() raise — shutdown must still return the robot to
        # kPrepare (standing), never leave it in kWalking.
        self.act._client.Move = mock.Mock(side_effect=RuntimeError("link down"))
        self.act.shutdown()
        self.assertIn(_FakeRobotMode.kPrepare, self.act._client.mode_calls)
        self.assertNotIn(_FakeRobotMode.kDamping, self.act._client.mode_calls)


if __name__ == "__main__":
    unittest.main()
