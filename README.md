# NaVILA → K1 Walking — VLM-Driven Humanoid Navigation in MuJoCo

Closed-loop demo: a pretrained vision-language model (NaVILA) sees through the
K1 humanoid's head camera, decides natural-language navigation actions, and a
PPO-trained velocity-tracking policy turns those actions into joint targets so
the K1 *physically walks* in MuJoCo (legs swing, feet step) — no kinematic
sliding of the floating base.

| third-person view | head camera (what NaVILA sees) |
|---|---|
| ![](docs/demo_third_person.mp4) | ![](docs/demo_head_camera.mp4) |

The video shows the instruction `walk to the red box \| turn right 90 deg \| walk forward`:
the K1 walks straight to the red box (heading-assist controller cancels policy drift),
the open-loop turn controller spins it 90° in place in 2 s, then it walks south in
the new heading.

## Architecture (one process, two threads)

NaVILA inference takes ~400 ms; the walking policy needs hard 50 Hz deadlines.
They cannot run sequentially in the same loop — the robot would freeze for a
second every iteration and fall.

```
main thread:                            VLM thread:
  physics + walking policy at 50 Hz      pull latest 8-frame buffer snapshot
  render head + scene cams at 30 Hz      run NaVILA generate() on the snapshot
  push head frames into ring buffer  →   parse "move forward 75 cm" → (vx, vy, vyaw)
  read shared (vx, vy, vyaw) command  ←  publish under lock
```

The walking policy is the K1 velocity-tracking policy from `booster_train`
(Booster-Gym / NaVILA observation layout):
`cmd(3) | gait_phase(2) | gravity(3) | ang_vel(3) | joint_pos_rel(12) | joint_vel(12) | last_action(12)`
× 5-frame term-major history → **235→12 MLP** → 12 leg targets scattered into
the K1's 22-DoF default pose.

## Multi-step instructions

Chain sub-steps with `|`, `;`, or `then`. Each sub-step is fed to NaVILA
verbatim and terminates on the **first** of:

| trigger | when | how it's detected |
|---|---|---|
| **yaw target** | turn complete (±5° tolerance) | auto-parsed from `turn (left\|right) N (deg\|rad)` |
| **proximity** | K1 within 1 m of a known target | auto-detected from `red box`/`blue box`/`green box` |
| **closest approach** | K1 has gotten ≥1.5 m close and now retreating | tracks `min_distance` over the sub-step |
| **NaVILA "stop"** | NaVILA explicitly emits stop | `parse_action()` |
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

- **Heading assist** (`--heading-assist`, default on) — when the active sub-step
  has a known proximity target, overlays
  `vyaw = clip(K_p · (target_bearing − robot_yaw))` on top of NaVILA's command.
  Cancels the slow policy drift NaVILA cannot compensate for. With assist off
  the K1 can drift ~1.5 m off-axis per 8 s of walking; with it on it stays
  within ±0.2 m.
- **Open-loop turn controller** (`--turn-controller`, default on) — for pure
  turn sub-steps NaVILA stubbornly emits "move forward" most of the time.
  Bypasses NaVILA's velocity, drives `vx = vy = 0`,
  `vyaw = clip(K_p · (yaw_target − yaw_unwrap))` with a `--turn-min-vyaw`
  floor (the discrete walking gait can't execute very small yaw commands).
  Result: a 90° turn takes ~2 s instead of timing out at 25 s.

The controller in charge each tick is shown in the on-frame HUD as
`+turn-controller` / `+heading` / unmarked.

## Usage

```bash
# the full closed-loop walking demo
python navila_k1_walking_loop.py \
  --instruction "walk to the red box | turn right 90 deg | walk forward" \
  --per-step-time 25 --save-video out
```

Output: `out/scene_view.mp4` (third-person) and `out/head_view.mp4` (NaVILA's
input view), both at 30 fps with a live HUD showing active sub-step,
termination metric, and applied-vs-VLM commands.

For pipeline debugging without burning the VLM:
```bash
python navila_k1_walking_loop.py --no-vlm --debug-vx 0.4 --max-sim-seconds 6 --save-video out
```

### Key flags

| flag | default | what it does |
|---|---|---|
| `--instruction <str>` | required | Multi-step instruction; `\|` / `;` / `then` separate sub-steps |
| `--per-step-time <s>` | 25 | Per-sub-step time-limit backstop |
| `--proximity-threshold <m>` | 1.0 | Sub-step done within this distance of named target |
| `--heading-assist / --no-heading-assist` | on | Yaw P-controller toward proximity target |
| `--heading-kp <r/s per r>` | 1.5 | Heading-assist gain |
| `--turn-controller / --no-turn-controller` | on | Bypass NaVILA for pure-turn sub-steps |
| `--turn-kp <r/s per r>` | 2.0 | Turn-controller gain |
| `--turn-min-vyaw <r/s>` | 0.30 | Floor on \|vyaw\| so the gait actually rotates |
| `--turn-tolerance-deg <°>` | 5.0 | Yaw target ±tolerance for sub-step done |
| `--vx-max / --vyaw-max` | 0.6 / 0.6 | Hard caps on commanded velocity |
| `--save-video <dir>` | — | Write `scene_view.mp4` + `head_view.mp4` to dir |
| `--no-vlm --debug-vx <m/s>` | — | Bypass NaVILA, hold a constant cmd (pipeline test) |

## Setup notes

This script is the **glue** for several upstream pieces. To actually run it
you need:

1. The `navila` conda env (Python 3.10, torch 2.7.1+cu128, mujoco 3.8, NaVILA's
   `llava` package installed editable).
2. **NaVILA repo + checkpoint** at `~/Projects/k1_research/booster/NaVILA/` with
   `checkpoints/navila-llama3-8b-8f/`. See [NaVILA on GitHub](https://github.com/NVIDIA/NaVILA).
3. **Booster K1 assets** at `~/Projects/k1_research/booster/booster_assets/` —
   the K1 22-DoF MJCF and meshes.
4. **Booster K1 velocity-tracking policy** registered in `booster_deploy` as
   the `k1_velocity` task (PPO-trained at 50 Hz under
   `Booster-K1-Velocity-v0`; observation layout matches NaVILA / Booster-Gym).
   The deploy task lives at `booster_deploy/tasks/locomotion/k1_velocity.py`.

The path constants live at the top of `navila_k1_walking_loop.py`:
```python
_K1RES = Path.home() / "Projects" / "k1_research"
_NAVILA_REPO        = _K1RES / "booster" / "NaVILA"
_BOOSTER_ASSETS_SRC = _K1RES / "booster" / "booster_assets" / "src"
_BOOSTER_DEPLOY     = _K1RES / "booster" / "booster_deploy"
```
Adjust if your layout differs.

The `MUJOCO_GL=egl` env var is set automatically before mujoco import — the
GLFW backend conflicts with torch's CUDA init in the same process.

## Files

| file | purpose |
|---|---|
| [`navila_k1_walking_loop.py`](navila_k1_walking_loop.py) | the closed-loop walking demo (this README's subject) |
| [`navila_k1_bridge.py`](navila_k1_bridge.py) | NaVILA loader, prompt builder, action regex parser, real-robot SDK sender stub |
| [`navila_mujoco_loop.py`](navila_mujoco_loop.py) | older sliding demo — kinematically translates the floating base; useful for VLM-perception sanity checks without the walking policy |
| [`test_navila.py`](test_navila.py) | smoke test: load NaVILA, run one inference on a single image |

## License

MIT.
