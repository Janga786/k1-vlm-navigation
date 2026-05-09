# Validation Pyramid — pre-flight before the K1 actually moves

The real-robot path uses the K1's **built-in walker** (`B1LocoClient.Move`),
which Booster has battle-tested across 700+ robots. The trained velocity
policy in `booster_deploy/tasks/locomotion/k1_velocity.py` is for
sim2sim only and is NOT in this loop. So the only real-world unknown is:
**does NaVILA produce sensible commands from real ZED camera images?**

NaVILA was trained on real indoor video (R2R / RxR / YouTube /
Matterport3D) — it is *probably* better-behaved on real ZED frames than
on the synthetic MuJoCo renders we've been throwing at it. The MuJoCo
renders are actually the harder test for NaVILA. So the validation
strategy is:

1. **Test the planner+parser+SDK wrapper exhaustively offline** so we
   know the only thing that can fail in deploy is NaVILA's perception.
2. **Validate NaVILA on the real ZED in dry-run mode** before any
   motion. Watch what it would tell the robot to do, with the floor
   clear and the operator within reach.
3. **Then go live**, with conservative speed caps and the kill switch
   ready.

The sections below are climbing rungs of the pyramid — each step must
pass before you go to the next.

---

## L0  Software present

```bash
# in the navila env (which is the deployment env)
/home/janga/miniconda3/envs/navila/bin/python -c "
import torch, mujoco, PIL, cv2, numpy as np
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('mujoco', mujoco.__version__)
print('cv2', cv2.__version__)
"
```
Expect: torch with CUDA, mujoco ≥ 3.x, cv2.

```bash
# llava (NaVILA's own package) — needed only at deploy time, not for tests
/home/janga/miniconda3/envs/navila/bin/python -c "
import sys; sys.path.insert(0, '/home/janga/Projects/k1_research/booster/NaVILA')
from llava.model.builder import load_pretrained_model
print('llava ok')
"
```

```bash
# K1 SDK — needed for --mode dry / live, NOT for --mode print
/home/janga/miniconda3/envs/navila/bin/python -c "
import booster_robotics_sdk_python as sdk
print('SDK ok:', sdk.__version__ if hasattr(sdk, '__version__') else 'present')
"
```
If absent, `--mode print` still works (no SDK calls). To install on the
K1 NUC see Booster docs.

---

## L1  Unit tests — pure Python, no NaVILA, no SDK, no MuJoCo (~5 s)

```bash
cd ~/Projects/k1_research/experiments/navila
python -m unittest discover tests -v
```

**Expect: `Ran 58 tests in ~5s, OK`.**

Coverage:

| file | what it pins down |
|---|---|
| `tests/test_action_parser.py` | NaVILA "move forward 75 cm" / "turn left 30 deg" / "stop" → (vx, vy, vyaw) regex correctness; clipping; unparseable→stop |
| `tests/test_planner.py` | Multi-step splitter (`\|`, `;`, `then`); auto-detection of yaw/proximity targets; termination logic (yaw delta, proximity, closest-approach, time, stop); priority order; `wrap_pi`; `yaw_from_quat`; controller dispatch (TURN/HEAD/VLM tags); min-vyaw floor; tag fallback when no pose source |
| `tests/test_sdk_dryrun.py` | Mocks `booster_robotics_sdk_python` and verifies: PrintActuator never touches the SDK; DryRunActuator never calls Move(); LiveActuator initialises with kWalking, sends Move() at SEND_HZ, watchdog zeros stale commands, shutdown always calls kDamping (even if Move() fails — robot must never be left in kWalking) |

If any test fails, **stop**. Do not proceed up the pyramid.

---

## L2  MuJoCo end-to-end with the trained policy (~20 s)

This is the hardest test for NaVILA's perception (synthetic renders are
out of distribution). If the script even just runs, the deploy script
will run too.

```bash
cd ~/Projects/k1_research/experiments/navila
/home/janga/miniconda3/envs/navila/bin/python navila_k1_walking_loop.py \
    --instruction "walk to the red box | turn right 90 deg | walk forward" \
    --per-step-time 25 --max-sim-seconds 60 \
    --save-video out_l2
```

Pass criteria:
- No exceptions (NaVILA loads, MuJoCo opens, controller runs).
- `[plan] sub-step 1/3 done: reached target (d≈0.99m)` fires within ~10 s.
- `[plan] sub-step 2/3 done: yaw target reached (Δ≈-90°)` fires within ~3 s of starting that sub-step.
- `[plan] all sub-steps complete` prints, then `final pose` prints.
- `out_l2/scene_view.mp4` shows the K1 walking, not sliding.

If the K1 falls or the planner hangs:
- Check the matched PD gains (`booster_deploy/tasks/locomotion/k1_velocity.py` line ~165 — `joint_stiffness`/`joint_damping`).
- Re-run with `--no-vlm --debug-vx 0.4 --max-sim-seconds 6` to confirm the *walking* layer is healthy independent of NaVILA.

---

## L3  Real-robot dry-run (with the K1 powered up but DOES NOT MOVE)

Goal: validate that NaVILA produces sensible commands from the real ZED
camera, and that the SDK wrapper is correctly initialised, **without
the robot moving**.

```bash
# Connect the laptop to the K1's network. Confirm reachable:
ping -c 2 192.168.0.10                 # or whatever --net is

# Robot should be in kPrepare (sitting). Check the control panel.

cd ~/Projects/k1_research/experiments/navila
/home/janga/miniconda3/envs/navila/bin/python navila_k1_realrobot.py \
    --mode dry \
    --image-source zed \
    --net 192.168.0.10 \
    --instruction "walk forward, then turn right 90 deg, then walk forward"
```

Watch the console for ~30 s. **What you should see**:
- `[image] ZED open at VGA/30fps` — camera grab works.
- `[VLM] NaVILA ready.` — model loaded.
- `[actuator/dry] B1LocoClient initialised. NOT switching to kWalking. Move() calls will be logged, not sent.`
- A stream of `[VLM #NNN]` lines printing what NaVILA decided.
- A stream of `[actuator/dry] would send Move(+0.X, +0.0, +0.X) (NOT SENT)`.

**Pass criteria**:
- The VLM commands look reasonable for what the K1 sees (move forward when
  there's free space ahead, turn when the goal is to the side).
- The actuator log NEVER says `Move(...)` was sent — only logged.
- The robot stays in kPrepare the whole time (no leg motion).
- The watchdog never fires unexpectedly (no continuous zero-cmd warnings
  unless the planner intends them).

**If any of these fail, fix before going live.** In particular, if NaVILA
says "stop" within the first frame, it's probably misreading the scene —
check the ZED frame is right-side up and not blank.

---

## L4  Real-robot live (the K1 actually walks)

**Pre-flight checklist (human responsibility)**:

- [ ] Floor clear of people, cables, and trip hazards in a ≥ 4 × 4 m
      area centered on the K1's start position.
- [ ] No fragile objects (TVs, glassware) within fall radius (~1.5 m).
- [ ] Operator has a hand on the kill-switch / e-stop.
- [ ] Battery > 30%.
- [ ] L1 + L2 + L3 all pass.
- [ ] First instruction is **conservative**: `walk forward 1 m | stop`,
      not the multi-target stress test.
- [ ] You know how to abort:
      - **Kill switch** on the K1 itself (stops actuators immediately).
      - **Ctrl-C** in the terminal (LiveActuator.shutdown sends Move(0,0,0)
        and switches to kDamping; takes ~0.5 s).
      - **Power button** if both above fail.

**Conservative caps** (the realrobot.py defaults):
- `--vx-max 0.4` (vs sim default 0.6)
- `--vy-max 0.15`
- `--vyaw-max 0.4`
- `--watchdog-seconds 1.5` (zero cmd if planner update is older than this)
- `--send-hz 20`

**Conservative first run**:
```bash
/home/janga/miniconda3/envs/navila/bin/python navila_k1_realrobot.py \
    --mode live \
    --image-source zed \
    --net 192.168.0.10 \
    --instruction "walk forward 1 meter" \
    --per-step-time 8 --max-sim-seconds 20 \
    --vx-max 0.3 --vyaw-max 0.3
```

If the K1 walks 1 m and stops cleanly, you've validated the entire
stack. **Then** scale up to multi-step instructions.

---

## What can still go wrong (and what to do)

| symptom | likely cause | fix |
|---|---|---|
| K1 walks forward but VLM never says stop | NaVILA misreading the scene OR target-recognition failure | Tighten the instruction: include a recognizable visual landmark NaVILA was trained on ("walk to the chair", not "walk to the red box"). Or rely on `--per-step-time` backstop. |
| K1 keeps drifting laterally | The built-in walker has its own residual drift. Or NaVILA isn't emitting yaw corrections | Enable `--heading-assist` IF you have a pose source plugged into `PoseSource.read()`. Without odometry, the planner can only do time + NaVILA-stop termination. |
| Turn sub-step takes way too long | NaVILA stubbornly says "move forward" instead of turning. The default turn-controller bypasses NaVILA but needs IMU yaw — without a pose source it falls back to NaVILA-only. | Wire `PoseSource` to read IMU yaw from `LowState` topic. |
| Watchdog warns "stale command" repeatedly | NaVILA inference is taking > 1.5 s | Bump `--watchdog-seconds 2.5` OR lighten NaVILA (smaller resolution input) |
| K1 stops mid-walk and won't restart | The actuator is in kDamping after a planner shutdown | Cycle the K1 back to kPrepare manually before re-running |
| `[actuator/live] Move() failed` after the run starts | Lost network or robot turned off | Check the ethernet cable and `ping`. The watchdog will pin the robot at 0 cmd. |

---

## Architecture diagram for reference

```
                ┌──────────────────────────────────────┐
                │           ImageSource                │
                │  zed | mjpeg | dir | static          │
                └──────────────────┬───────────────────┘
                                   │ PIL.Image
                ┌──────────────────▼───────────────────┐
                │         VLMRunner (thread)           │
                │   - 8-frame ring buffer              │
                │   - NaVILA generate() ~400 ms        │
                │   - parse_action regex               │
                └──────────────────┬───────────────────┘
                                   │ (vx, vy, vyaw)
                                   │ + stop_event
                ┌──────────────────▼───────────────────┐
                │       Multi-step planner             │
                │   - parse_substeps                   │
                │   - check_termination per tick       │
                │     (yaw target | proximity |        │
                │      closest approach | stop |       │
                │      time)                           │
                │   - apply_controllers                │
                │     (TURN | HEAD | VLM passthrough)  │
                └──────────────────┬───────────────────┘
                                   │ (vx, vy, vyaw)
                ┌──────────────────▼───────────────────┐
                │             Actuator                 │
                │   print | dry | live                 │
                │   live = SDK Move() at SEND_HZ +     │
                │     watchdog + kDamping on shutdown  │
                └──────────────────────────────────────┘
                                   │
                                   ▼
                  Real K1 (built-in walker)  OR
                  MuJoCo + trained policy (separate script)
```

The MuJoCo path uses `navila_k1_walking_loop.py` (trained policy +
physics). The real-robot path uses `navila_k1_realrobot.py` (built-in
walker). Both share the planner, controllers, parser, and VLMRunner via
`navila_k1_core.py`.
