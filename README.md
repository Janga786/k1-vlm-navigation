# NaVILA → K1 — VLM-Driven Humanoid Navigation

Closed-loop demos: a pretrained vision-language model (NaVILA) sees through the
K1 humanoid's head camera, emits natural-language navigation actions, and a
multi-step planner + classical controllers turn those actions into velocity
commands. Two execution paths share the same brain:

| script | who walks the K1 | use case |
|---|---|---|
| `navila_k1_walking_loop.py` | **trained PPO velocity policy** in MuJoCo | sim2sim development; harder NaVILA test (synthetic renders) |
| `navila_k1_realrobot.py` | **K1's built-in walker** (`B1LocoClient.Move`) | real-robot deploy; battle-tested by Booster across 700+ robots |

| third-person view | head camera (what NaVILA sees) |
|---|---|
| `docs/demo_third_person.mp4` | `docs/demo_head_camera.mp4` |

## Architecture (shared between both paths)

```
                ┌──────────────────────────────────────┐
                │           ImageSource                │
                │  zed | mjpeg | dir | static | mujoco │
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
                │   trained policy (sim) | print       │
                │   | dry SDK | live SDK               │
                └──────────────────────────────────────┘
```

The shared brain — VLMRunner, planner, controllers — lives in
**[`navila_k1_core.py`](navila_k1_core.py)** and is exhaustively unit-tested
in **[`tests/`](tests/)**.

## Multi-step instructions

Chain sub-steps with `|`, `;`, or `then`. Each sub-step terminates on the
**first** of:

| trigger | when | how it's detected |
|---|---|---|
| **yaw target** | turn complete (±5° tolerance) | auto-parsed from `turn (left\|right) N (deg\|rad)` |
| **proximity** | K1 within 1 m of a known target | auto-detected from `red box`/`blue box`/`green box` |
| **closest approach** | K1 has gotten ≥1.5 m close and now retreating | tracks `min_distance` |
| **NaVILA "stop"** | NaVILA emits stop | `parse_action()` |
| **time limit** | per-sub-step backstop (default 25 s) | wall-clock |

Examples:

```
"navigate to the red box"
"walk to the red box | turn right 90 deg | walk forward"
"walk to the red box; turn left 45 degrees; walk to the blue box"
"walk forward until reaching the red box, then turn right 90 deg, then walk forward"
```

## Inner-loop controllers

NaVILA's atomic actions are coarse and biased toward "move forward". Two
classical controllers ride underneath, both gated by the parsed sub-step type:

- **Heading assist** — proximity sub-steps overlay
  `vyaw = K_p · (target_bearing − robot_yaw)` on top of NaVILA's command.
  Cancels the slow gait drift NaVILA cannot compensate for. Without assist
  the K1 drifts ~1.5 m off-axis per 8 s of walking; with it, ±0.2 m.
- **Open-loop turn controller** — pure-turn sub-steps **bypass NaVILA**
  entirely (NaVILA stubbornly emits "move forward" most of the time).
  Drives `vx = vy = 0, vyaw = clip(K_p · (yaw_target − yaw_unwrap))`
  with a `--turn-min-vyaw` floor and ±5° termination tolerance. A 90°
  turn takes ~2 s instead of timing out at 25 s.

The active controller is shown in the on-frame HUD and in the console:
`applied[TURN] / [HEAD] / [VLM ]`.

## Real-robot path (`navila_k1_realrobot.py`)

Three modes, three image sources, **conservative defaults for hardware**:

```bash
# offline NaVILA-on-recorded-frames sanity (no SDK, no robot):
python navila_k1_realrobot.py --mode print \
    --image-source dir --image-dir ./recorded_zed_frames \
    --instruction "navigate to the red chair"

# all-up dry-run on the actual robot, NO MOTION:
python navila_k1_realrobot.py --mode dry \
    --image-source zed --net 192.168.0.10 \
    --instruction "walk forward 3 meters | turn left 90 deg | walk forward"

# live deploy (REQUIRES the floor cleared):
python navila_k1_realrobot.py --mode live \
    --image-source zed --net 192.168.0.10 \
    --instruction "walk to the chair | turn right 90 deg | walk forward"
```

**Mode behaviour**:

| mode | SDK init | Move() sent | mode change | use |
|---|---|---|---|---|
| `print` | none | logged only | none | NaVILA validation, offline replay |
| `dry`   | yes  | logged only | **none** (stays in kPrepare) | pre-flight on the real robot |
| `live`  | yes  | yes at SEND_HZ | kWalking → kDamping | actual deploy |

The LiveActuator runs a background sender at 20 Hz with a **1.5 s watchdog**
that zeros the cmd if the planner's update is older than that, so a hung
NaVILA can't carry the robot into a wall. On shutdown it always switches
the K1 back to **kDamping**, even if the zero-Move() call fails.

Default velocity caps for `realrobot` are conservative (`vx_max=0.4`,
`vy_max=0.15`, `vyaw_max=0.4`); the MuJoCo loop uses `0.6/0.3/0.6`. Override
with `--vx-max` etc.

See **[VALIDATION.md](VALIDATION.md)** for the pre-flight pyramid before
going live.

## MuJoCo path (`navila_k1_walking_loop.py`)

```bash
python navila_k1_walking_loop.py \
  --instruction "walk to the red box | turn right 90 deg | walk forward" \
  --per-step-time 25 --save-video out
```

Output: `out/scene_view.mp4` + `out/head_view.mp4` at 30 fps with a live
HUD. For pipeline debugging without burning the VLM:

```bash
python navila_k1_walking_loop.py --no-vlm --debug-vx 0.4 --max-sim-seconds 6 --save-video out
```

The walking policy is the K1 velocity-tracking policy (Booster-Gym /
NaVILA observation layout): 235→12 MLP, 50 Hz inference, scattered into
the K1's 22-DoF default pose. Lives at `booster_deploy/tasks/locomotion/k1_velocity.py`.

### Key flags (both scripts share most of these)

| flag | default | what it does |
|---|---|---|
| `--instruction <str>` | required | Multi-step instruction; `\|` / `;` / `then` split sub-steps |
| `--per-step-time <s>` | 25 | Per-sub-step time-limit backstop |
| `--proximity-threshold <m>` | 1.0 | Sub-step done within this distance of target |
| `--heading-assist / --no-heading-assist` | sim:on, real:off | Yaw P-controller toward target (needs pose) |
| `--heading-kp <r/s per r>` | 1.5 | Heading-assist gain |
| `--turn-controller / --no-turn-controller` | on | Bypass NaVILA for pure-turn sub-steps |
| `--turn-kp <r/s per r>` | 2.0 | Turn-controller gain |
| `--turn-min-vyaw <r/s>` | 0.30 | Floor on \|vyaw\| so the gait actually rotates |
| `--turn-tolerance-deg <°>` | 5.0 | Yaw target ±tolerance for sub-step done |
| `--vx-max / --vyaw-max` | sim:0.6/0.6, real:0.4/0.4 | Hard caps on commanded velocity |
| `--save-video <dir>` | — | (sim only) write `scene_view.mp4` + `head_view.mp4` |
| `--no-vlm --debug-vx <m/s>` | — | (sim only) bypass NaVILA, hold a constant cmd |
| `--mode {print,dry,live}` | print | (real only) actuator backend |
| `--image-source {zed,mjpeg,dir,static}` | static | (real only) image input |

## Validation

```bash
# 58 unit tests: parser regex, sub-step splitter, termination logic,
# closest-approach, controller dispatch, helpers, mocked SDK actuators.
# No NaVILA, no MuJoCo, no SDK required.
python -m unittest discover tests -v
```

Pass criterion: `Ran 58 tests in ~5s. OK`.

For everything beyond the unit tests (sim end-to-end, real-robot dry-run,
real-robot live), see **[VALIDATION.md](VALIDATION.md)** — climbs the
pyramid from offline tests all the way to a kill-switch-ready live deploy.

## Files

| file | purpose |
|---|---|
| [`navila_k1_core.py`](navila_k1_core.py) | shared brain: SubStep, parse_substeps, VLMRunner, controllers, helpers |
| [`navila_k1_walking_loop.py`](navila_k1_walking_loop.py) | MuJoCo + trained velocity policy (sim2sim) |
| [`navila_k1_realrobot.py`](navila_k1_realrobot.py) | K1 SDK (B1LocoClient) — print / dry / live modes |
| [`navila_k1_bridge.py`](navila_k1_bridge.py) | NaVILA loader, prompt builder, action regex parser |
| [`navila_mujoco_loop.py`](navila_mujoco_loop.py) | older sliding demo (kinematic, no walker) |
| [`test_navila.py`](test_navila.py) | manual NaVILA smoke test on a single image |
| [`tests/`](tests/) | 58 unit tests — runnable in any Python 3.10+ env |
| [`VALIDATION.md`](VALIDATION.md) | pre-flight pyramid for real-robot deploy |
| [`docs/`](docs/) | demo MP4s |

## Setup notes

This script is the **glue** for several upstream pieces. To run end-to-end
you need:

1. The `navila` conda env (Python 3.10, torch 2.7+, mujoco 3.x, NaVILA's
   `llava` package installed editable).
2. **NaVILA repo + checkpoint** at
   `~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f`.
   See [NaVILA on GitHub](https://github.com/NVIDIA/NaVILA).
3. **Booster K1 assets** at `~/Projects/k1_research/booster/booster_assets/`
   (sim path only).
4. **Booster K1 velocity-tracking policy** registered in `booster_deploy`
   as the `k1_velocity` task (sim path only).
5. **`booster_robotics_sdk_python`** for `--mode dry/live` of the real-robot
   path (skip if you only want the offline / sim flow).

`MUJOCO_GL=egl` is set automatically before mujoco import — GLFW conflicts
with torch's CUDA init.

## License

MIT.
