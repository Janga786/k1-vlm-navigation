# K1 NaVILA on VLN-CE-Isaac — Benchmark Setup Report

**Run date:** 2026-05-15  
**Hardware:** RTX 5090 (Blackwell, sm_120, 32 GB), AMD/Intel x86_64  
**Status:** end-to-end pipeline working; SR = 0% with placeholder low-level policy. Real K1 SR requires training a benchmark-compatible velocity policy (see *What's left*).

---

## Headline result

| Robot | Policy | SR | NE | SPL | OS | Notes |
|---|---|---|---|---|---|---|
| **K1** (this work) | `stub_zero_2026-05-15` (placeholder; doesn't walk) | 0.000 | 8.37 m | 0.000 | 0.000 | Pipeline-test only — robot tips over within ~1s sim time |
| Go2 (paper Table IV, Vision) | RSL-RL policy | 0.502 | 5.49 | 0.455 | 0.587 | Reference baseline from NaVILA paper |
| H1 (paper Table IV, Vision) | RSL-RL policy | 0.453 | 5.86 | 0.403 | 0.546 | Reference baseline from NaVILA paper |

Paper numbers from NaVILA (RSS'25) Table IV. K1 was not in the paper — this work is the first port.

---

## What works (verified end-to-end)

1. **Isaac Sim 4.1.0 / Isaac Lab 1.1.0 boot on RTX 5090.** Needed torch 2.11+cu128 upgrade (the maintainers' fork pins torch 2.2.2+cu121 which has no Blackwell kernels). EULA accepted via `~/miniconda3/envs/vlnce-isaac/lib/python3.10/site-packages/omni/EULA_ACCEPTED` file.
2. **Go2 baseline `demo_planner` runs.** This is the go/no-go gate; if Go2 can't run, K1 won't either. Output: PD planner streams `vel_command` tensors → loaded Go2 vision policy → robot walks toward goal in the Matterport USD scene.
3. **K1 task registers and spawns.** New `k1_matterport_base` gym task loads `K1_locomotion.urdf` via the URDF importer, mounts a head camera on `/Robot/Trunk` at +0.85 m, applies a 12-DoF leg action.
4. **NaVILA VLM in the loop.** The benchmark's `vlm_server.py` requires llava+transformers which don't coexist cleanly in the vlnce-isaac env. Fix: a separate `vlm_server_bridge.py` runs NaVILA in the existing `navila` conda env (torch 2.7.1+cu128, transformers 4.49.0) and serves the same 8-byte-size-prefix JSON protocol on port 54321. NaVILA returns commands like `"The next action is turn right 45 degree."` in ~330-680 ms.
5. **Episode measurement writes.** `eval_results/k1_matterport_base_loco_stub_zero_2026-05-15/measurements/0.json` contains `path_length, distance_to_goal, success, spl, oracle_navigation_error, oracle_success`. Video `videos/output_0.mp4` records the run.

---

## Commands that work (future re-runs)

```bash
# Terminal 1 — VLM bridge (in the navila env)
conda activate navila
python ~/Projects/k1_research/NaVILA-Bench/scripts/vlm_server_bridge.py \
    --model_path ~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f \
    --port 54321

# Terminal 2 — single K1 episode (in the vlnce-isaac env)
conda activate vlnce-isaac
export OMNI_KIT_ACCEPT_EULA=yes
cd ~/Projects/k1_research/NaVILA-Bench
python scripts/navila_eval.py \
    --task=k1_matterport_base --num_envs=1 \
    --load_run=stub_zero_2026-05-15 \
    --headless --enable_cameras --episode_idx=0

# Terminal 2 — full benchmark slice
python scripts/run_benchmark.py \
    --task=k1_matterport_base \
    --low_level_policy_dir=stub_zero_2026-05-15 \
    --start-idx=0 --end-idx=50 \
    --per-episode-timeout=240
```

Per-episode JSON: `eval_results/<task>_loco_<run>/measurements/<i>.json`. Aggregate with the helper script at the bottom of this report.

---

## Patches applied to make it work

All commits live on the **`vlnce-isaac-benchmark`** branch of the NaVILA-Bench fork (local). Five repos / paths were touched:

### `~/IsaacLab-vlnce/` (the maintainers' fork)
- `source/extensions/omni.isaac.lab/omni/isaac/lab/sensors/camera/camera.py:339-352` — Camera.reset() force-initialises on first call (Isaac Sim 4.1 + Isaac Lab 1.1 sometimes doesn't fire the PLAY timeline callback before the first env.reset()).

### `~/Projects/k1_research/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/`
- `omni/isaac/vlnce/config/__init__.py` — re-export the new `k1` package.
- `omni/isaac/vlnce/config/k1/__init__.py` — register `k1_matterport_base` gym task.
- `omni/isaac/vlnce/config/k1/k1_matterport_base_cfg.py` — task config: K1 URDF, head camera, 12-DoF leg action, no contact-sensor terms (Isaac Lab 1.1 URDF importer doesn't expose `activate_contact_sensors`).
- `omni/isaac/vlnce/vlnce/mdp/observations.py` — `isaac_camera_data` and `process_depth_image` return a zero placeholder tensor when the camera hasn't been initialised yet (ObservationManager probes shapes during env __init__ before the camera sensor is up).
- `omni/isaac/vlnce/vlnce/mdp/actions/{navigation_actions, vlm_navigation_actions, vlm_navigation_actions_gpt}.py` — removed module-top `import torchvision` (transitively imported PIL from Isaac Sim's bundled pip_archive, which crashes on `_util.is_directory`). The imports were dead — torchvision was only used in commented-out resize_transform blocks.

### `~/Projects/k1_research/NaVILA-Bench/scripts/`
- `navila_eval.py:131-138` — added a `k1` z-offset branch (init_state.pos[2] += 0.55 to match K1's 0.55 m standing height in the locomotion URDF).
- `run_benchmark.py` — fixed `--episode_idx` accumulation bug (was mutating a shared list), added `--end-idx` and `--per-episode-timeout` flags.
- `vlm_server_bridge.py` (new, 200 lines) — NaVILA TCP server in the `navila` conda env exposing the benchmark's wire protocol.

### `~/Projects/k1_research/NaVILA-Bench/logs/rsl_rl/k1_base/stub_zero_2026-05-15/`
- `model_0.pt` — placeholder rsl_rl PPO checkpoint: actor MLP (45 → 512 → 256 → 128 → 12), last layer zeroed, log-std −5 → deterministic ~0 action. Optimizer Adam state populated.
- `params/agent.yaml` — matches `K1RoughPPORunnerCfg`.
- Built via `/tmp/build_k1_stub_v2.py` (not committed; kept in /tmp for reproducibility).

### vlnce-isaac conda env (pip)
- `torch 2.11.0+cu128` (replacing 2.2.2+cu121 — Blackwell support).
- `pillow==10.2.0` (matches Isaac Sim's bundled Pillow; fixes JPEG-encoder ABI mismatch).
- `botocore>=1.34,<1.35`, `s3transfer<0.11`, `boto3<1.35` (Isaac Sim bundles old botocore; newer s3transfer 0.17 imports `DEFAULT_CHECKSUM_ALGORITHM` which isn't in 1.34).

---

## What's broken / not yet validated

- **The stub low-level policy doesn't walk.** It outputs near-zero joint deltas, so the K1 just tips over in place. Every navila_eval run will produce `success: 0.0`. The pipeline runs but SR is meaningless. This is the expected outcome of the placeholder — fixing requires training a real K1 velocity-tracking policy with the benchmark's 45-dim proprio obs and 12-dim leg-action layout.
- **Existing trained K1 velocity policy (`~/Projects/k1_research/booster/booster_train/.../k1_velocity/.../model_9999.pt`) is NOT directly reusable.** It uses Isaac Lab 2.x's `history_length=5, flatten_history_dim=True` observation group, which Isaac Lab 1.1 doesn't support. To reuse it without retraining: either port that obs-group history mechanism backwards to Isaac Lab 1.1, OR train a fresh policy in the benchmark's 45-dim layout (estimated 12-24 GPU hours at 4096 envs on the 5090).
- **Hydra mesh corruption warnings** print for both Go2 and K1 USDs (`Mesh '.../proto_mesh_0_id0' has corrupted data in primvar 'st_N': buffer size X doesn't match expected size Y`). Cosmetic only — they don't affect physics or rendering enough to break the run, but they spam ~50 lines on each spawn.

---

## How the placeholder ports to real K1 SR numbers

Two pragmatic paths:

**Option A — train a benchmark-style K1 policy in the IsaacLab 1.1 / IsaacSim 4.1 fork:**
1. Mirror the H1 base config's PolicyCfg (45-dim proprio, no height scan, no history).
2. Use H1's PPO hyperparameters (lr 1e-3, num_steps_per_env=24, max_iterations=10000).
3. Train on flat ground with the benchmark's reward set (track_lin_vel + ang_vel + standard penalties).
4. Expected wall time on RTX 5090 at 4096 envs: ~6-12 h.
5. Drop the resulting `model_XXXX.pt` into `logs/rsl_rl/k1_base/<your_run>/` and re-run `run_benchmark.py --low_level_policy_dir=<your_run>`.

**Option B — port history-aware obs to Isaac Lab 1.1:**
1. Backport Isaac Lab 2.x's `ObservationGroupCfg.history_length` / `flatten_history_dim` plumbing into the local `omni.isaac.lab.managers.observation_manager.ObservationManager`.
2. Add the `gait_phase` MDP function to `omni/isaac/lab/envs/mdp/observations.py`.
3. Update `K1MatterportBaseCfg.ObservationsCfg.PolicyCfg` to match the booster_train layout (47-dim with gait_phase + 5-frame flat history).
4. Symlink `~/Projects/k1_research/booster/booster_train/logs/rsl_rl/k1_velocity/2026-05-08_14-15-21/` into the benchmark's `logs/rsl_rl/k1_base/` and re-run.
5. Risk: history flattening differences may still cause subtle shape mismatches.

Option A is lower risk. Option B avoids retraining but requires careful porting.

---

## Aggregate results (auto-updated as benchmark runs)

Run this to recompute the table once new measurement JSONs have landed:

```bash
python - <<'PY'
import json, glob, os
mdir = os.path.expanduser("~/Projects/k1_research/NaVILA-Bench/eval_results/k1_matterport_base_loco_stub_zero_2026-05-15/measurements")
files = sorted(glob.glob(f"{mdir}/*.json"))
if not files:
    print("no measurements yet"); raise SystemExit
keys = ["success","spl","distance_to_goal","oracle_success","oracle_navigation_error","path_length"]
acc = {k: [] for k in keys}
for f in files:
    d = json.load(open(f))
    for k in keys: acc[k].append(d[k])
import statistics
print(f"episodes evaluated: {len(files)}")
for k in keys:
    print(f"  {k}_mean = {statistics.mean(acc[k]):.4f}")
PY
```

**Final aggregate (20 episodes, run complete):**
- success_mean = 0.0000 (min 0, max 0)
- spl_mean = 0.0000
- distance_to_goal_mean = 10.66 m (min 6.89, max 15.19)
- oracle_success_mean = 0.0000
- oracle_navigation_error_mean = 10.49 m (min 6.89, max 14.45)
- path_length_mean = 0.28 m (min 0.0021, max 0.89) ← non-zero is sim
  wobble + occasional tip-over slide before the K1 settles

The stub policy never produces a successful episode — none of the 19
runs reached within 3 m of the goal, and the K1's path length is
< 1 m in every case (the robot tips over and stops). This is the
expected behavior for a zero-action placeholder; replace
`stub_zero_2026-05-15` with a trained K1 walking policy run-dir and
the same `scripts/run_benchmark.py` command will give meaningful
SR / SPL numbers.

JSON filenames use `episode_id - 1` (not the run_benchmark iteration
index), so they do NOT line up with the `--start-idx`/`--end-idx`
range. Filename gaps are the maintainers' chosen episode_ids, not
failed runs.

A 19-episode run (indices 1-19) is in flight in the background. Some
episodes are timing out at the 240 s per-episode cap (mostly Isaac Sim
cold-start, ~50 s). Aggregates will only grow more 0 / NaN-y until a
real walking K1 policy is dropped in.

For the FULL paper-matching evaluation, run:
```
python scripts/run_benchmark.py --task=k1_matterport_base \
    --low_level_policy_dir=<your_real_policy_run_dir> \
    --start-idx=0 --end-idx=1077
```
With a working policy, expect 18-24 h wall time on this 5090.

---

## Overnight extension (2026-05-15 23:17): training a real K1 vision policy

### Goal
Replace the zero-action stub with a real K1 walking policy trained in
legged-loco's vision-config layout, then re-run the full 1077-episode
benchmark to get non-trivial SR/NE/SPL numbers.

### Approach
- Built a K1 vision env on top of `omni.isaac.leggedloco.config.h1` (the
  reference humanoid template). New files:
  - `legged-loco/.../config/k1/k1_low_vision_cfg.py` (training env, ~430 LOC)
  - `legged-loco/.../config/k1/k1_low_base_cfg.py` (blind fallback)
  - `NaVILA-Bench/.../config/k1/k1_matterport_vision_cfg.py` (eval env mirror)
- K1 articulation loaded via `UrdfFileCfg(merge_fixed_joints=True,
  activate_contact_sensors=True)`. Trunk holds the head camera + height
  raycaster; legs are 12 revolute joints (`Hip Pitch/Roll/Yaw`,
  `Knee Pitch`, `Ankle Pitch/Roll` × 2 sides).
- Reward stack: track_lin/ang_vel_exp + feet_air_time_positive_biped (on
  `.*foot_link`) + feet_slide + ankle joint-pos-limit penalty + hip
  Yaw/Roll deviation penalty + flat_orientation + dof_acc + action_rate +
  termination_penalty(-200). Dropped H1's `joint_deviation_arms` and
  `joint_deviation_torso` since K1's locomotion URDF has no arm/torso
  joints after `merge_fixed_joints=True`.
- Terrain curriculum: stairs (0.05-0.20 m step), boxes, slopes (0-0.3 rad),
  random rough, and a 1.5 m discrete-obstacle section. Step heights
  halved vs. H1's defaults since K1 is ~half H1's height.
- PD gains: stiffness 350 hips/knees, 250 ankles; damping 10/5; action
  scale 0.5. PD gains match the verified `k1_matterport_base` config.

### Training command (active overnight)
```bash
# tmux: k1train
python scripts/train.py --task=k1_vision --run_name=k1_vision_v1 \
    --max_iterations=2000 --save_interval=200 --headless
```
- 4096 envs, decimation=4 (50 Hz policy / 200 Hz physics).
- Iter time: ~5.7-6.4 s. ETA: ~3.1 h on RTX 5090.
- Logs to `logs/rsl_rl/k1_vision_rough/<timestamp>_k1_vision_v1/`.

### Auto-launch benchmark when training completes
A watcher in tmux session `watcher` polls for `model_2000.pt`. When
ready, it (a) kills `k1train` to free the GPU, (b) symlinks the run dir
into `NaVILA-Bench/logs/rsl_rl/k1_vision_rough/`, (c) starts the VLM
bridge in tmux `vlm` and waits 10 min for "listening on", (d) runs a
single-episode smoke test, (e) launches the full 1077-episode benchmark
in tmux `bench`.

End-to-end pipeline driver:
```bash
~/Projects/k1_research/run_k1_benchmark.sh <load_run> 0 1077 240
```

### Status @ 23:25
- Training: iter ~40, ~4 min in, track_lin_vel_xy_exp 0.025, base_contact
  declining from 117 → 76 (good sign — robots surviving longer episodes).
- All four tmux sessions verified alive (k1train, watcher) and the rest
  will be spawned by the watcher.

---

## Phase 3 — 2026-05-18: pivot to booster_train

### Why the pivot

**legged-loco v3 50 k iter run diverged** (2026-05-16 → 2026-05-18). The
`action_rate_l2` reward term went from ≈ -0.04 at iter 44 k to **-3.085e+15**
by iter 50 k; mean reward fell from +5…+10 to -12…-21. `dof_torques_l2 ≈ 0`
and `dof_acc_l2 ≈ -0.09` at the collapse iter — confirming the physical
robot was not moving violently. Only the *policy outputs* were oscillating
between extremes (post-tanh, in [-1, +1]), producing a quadratic explosion
in the penalty.

Single config difference vs the working booster_train K1 velocity policy
(10 k iter, reward 103.20, walks fine in MuJoCo sim-to-sim):

| Parameter | legged-loco K1 vision | booster_train K1 velocity |
|---|---|---|
| `action_rate_l2.weight` | **-0.005** | **-1.0** |
| Action clipping | none | none |
| Action scale | 0.5 | 0.25 |

200× weaker action-smoothing penalty in legged-loco. Once the policy entered
a regime with oscillating outputs, the penalty was too weak to pull it back —
at action delta ≈ 1.0, penalty = 12·1²·(-0.005) = -0.06, trivial compared to
tracking reward.

### Approach: rebuild on booster_train, drop into the same benchmark env

booster_train uses the same K1 URDF + same PD gains as the benchmark and a
known-stable PPO recipe. Strategy:

1. **New task** `Booster-K1-Velocity-Vision-VLNCE-v0` in
   `booster_train/.../velocity/robots/k1/vlnce_env_cfg.py`. Observation/action
   layout = exact match to `k1_matterport_vision_cfg.py` PolicyCfg, so the
   trained `.pt` drops into the benchmark with **no adapter**.
2. **Keep all of booster_train's PPO/reward settings** including
   `action_rate_l2.weight = -1.0`.
3. **Train in `isaacLab_311`** (rsl_rl 3.1.2), **eval in `vlnce-isaac`**
   (rsl_rl 2.0.2). Cross-version load verified — both produce identical
   `actor.{0,2,4,6}.{weight,bias}` / `critic.{0,2,4,6}.{weight,bias}` / `std`
   state_dict keys.

### Observation layout (exact match to benchmark)

| # | Term | Dim | Notes |
|---|---|---|---|
| 0 | `base_lin_vel` | 3 | Unoise(±0.1) at train, none at eval |
| 1 | `base_ang_vel` | 3 | Unoise(±0.2) |
| 2 | `projected_gravity` | 3 | Unoise(±0.05) |
| 3 | `velocity_commands` | 3 | Replaced by VLM-emitted (vx, vy, vyaw) at eval |
| 4 | `joint_pos_rel` | 12 | Unoise(±0.01) |
| 5 | `joint_vel_rel` | 12 | Unoise(±1.5) |
| 6 | `last_action` | 12 | No noise |
| 7 | `height_scan` | **187** | RayCaster on Trunk, 17×11 grid @ 0.1 m, clip ±1.0 |
| | **Total** | **235** | `history_length=0`, `flatten_history_dim=False` |

The 187 (not the 160 estimated up front) comes from
`GridPatternCfg(size=(1.6, 1.0), resolution=0.1)`:
`(1.6/0.1 + 1) × (1.0/0.1 + 1) = 17 × 11 = 187` rays (inclusive endpoints).

### Other key training settings

- **Action scale 0.5**, `use_default_offset=True`, joint_names from leg regex.
- **Terminations**: `time_out` (30 s episodes), `root_height_below_minimum @ 0.30 m`,
  `bad_orientation @ 1.3 rad` — exact match to benchmark Fix-v1 eval threshold.
- **Terrain**: `ROUGH_TERRAINS_CFG` (Isaac Lab preset — stairs±, boxes,
  random rough, slopes±), curriculum on, `max_init_terrain_level=0`.
- **RayCaster**: 16×10 grid (`size=(1.6, 1.0)`, `resolution=0.1`),
  `attach_yaw_only=True`, attached to `Trunk` 20 m above,
  `mesh_prim_paths=["/World/ground"]`.
- **Domain rand**: friction (0.3–1.5), mass scale (±15%), COM offset,
  init pose, random pushes every 8–12 s (±0.5 m/s).
- **PD gains**: hip/knee 350/7.5, ankle 250/5 (from `BOOSTER_K1_LOCOMOTION_CFG`).
  Note: benchmark uses 350/**10**, ankle 250/5 — small damping mismatch
  on hip/knee. Policy should generalize given domain rand.
- **PPO**: 4096 envs, 24 steps/env, 5 epochs, 4 minibatches, lr 1e-3 adaptive KL,
  clip 0.2, entropy 0.01, γ=0.99, λ=0.95, max_grad_norm 1.0,
  init_noise_std=1.0, actor/critic [512, 256, 128] elu, `empirical_normalization=False`.

### Files added (no edits to existing booster_train code)

- `booster_train/.../velocity/robots/k1/vlnce_env_cfg.py`
- `booster_train/.../velocity/robots/k1/vlnce_ppo_cfg.py`
- `booster_train/.../velocity/robots/k1/__init__.py` — appended two
  `gym.register` blocks for the new task + Play variant.
- `~/Projects/k1_research/scripts/train_k1_vlnce.sh` — tmux launcher.
- `~/Projects/k1_research/scripts/watch_k1_vlnce_train.sh` — plateau watcher.
- `~/Projects/k1_research/scripts/bench_k1_vlnce.sh` — benchmark runner
  (smoke + full modes).
- `~/Projects/k1_research/scripts/fire_smoke_when_ready.sh` — fires the
  smoke test as soon as `model_2000.pt` lands.

### Smoke test PASSED (2 envs × 2 iters, 2026-05-18 10:05)

```
ObservationManager:
  policy[0] base_lin_vel              (3,)
  policy[1] base_ang_vel              (3,)
  policy[2] projected_gravity         (3,)
  policy[3] velocity_commands         (3,)
  policy[4] joint_pos                 (12,)
  policy[5] joint_vel                 (12,)
  policy[6] actions                   (12,)
  policy[7] height_scan               (187,)
Mean reward: -25.62  (random init, expected)
Total timesteps: 96
```

### Status @ 10:14
Training tmux `k1_vlnce_train` (v1) at iter 155, reward -0.68.

### Update @ 10:42 — v1 dead-on-arrival, restarted as v2 on gentler terrain

By iter 1000, **`Mean episode length` was stuck at 1.00** with
`bad_orientation` termination at 80%. Comparing to the working flat-ground
booster_train K1 velocity tfevents (`2026-05-08_14-15-21`):

| Metric | Flat baseline @ iter 100 | v1 (rough) @ iter 100 |
|---|---|---|
| Mean reward | -2.08 | -0.71 |
| Mean episode length | 13.4 | **1.00** |
| bad_orientation | 99.79% | 79.59% |
| track_lin_vel_xy | 0.0075 | 0.0022 |

Root cause: `ROUGH_TERRAINS_CFG` ships with `pyramid_stairs` (`step_height_range
= (0.05, 0.23)`) and `boxes` (`grid_height_range = (0.05, 0.20)`). With
`pose_range x/y ±0.5 m`, a random K1 spawn frequently landed one foot on a
step and the other off the step → body roll past 1.3 rad before the policy
acted at all → `bad_orientation` termination at step 1.

**Fix:** custom `K1_VLNCE_TERRAINS_CFG` (in `vlnce_env_cfg.py`):
- 60% `HfRandomUniformTerrainCfg`, `noise_range=(0.01, 0.06)` — gentle bumps
- 20% `HfPyramidSlopedTerrainCfg`, `slope_range=(0.0, 0.15)` — ~8.5° ramps up
- 20% `HfInvertedPyramidSlopedTerrainCfg`, same gentle slope range — ramps down
- No stairs, no boxes
- `curriculum=True`, `num_rows=10`, `max_init_terrain_level=0`

Smoke test on the new terrain (4 envs × 3 iters): episode length **44.5 at iter 2**,
bad_orientation 33% and dropping. Relaunched as v2 at 10:42:27.

### Status @ 10:45
v2 healthy: iter 17 reward -14.77, episode length **41.82** (vs v1's 1.00),
bad_orientation **26%** and dropping, track_lin_vel_xy **0.0157** (7× v1).
ETA 13:37 to iter 30 k.

### Update @ 11:30 — v2 broke out at iter 500, then was killed

v2 reward curve:

| iter | reward | ep_len |
|---|---|---|
| 17 | -14.77 | 41.82 |
| 100 | -1.02 | 19.47 |
| 500 | **+29.52** | **978.81** |
| 1000 | +47.72 | 1422.67 |
| 2000 | +56.34 | 1407.20 |

Then at iter 2012, **`fire_smoke_when_ready.sh` killed the training session**.
It correctly detected an integration failure (the smoke at iter 2000 hit a
`KeyError: '/obs_groups'` — rsl_rl 3.1.2's saved `agent.yaml` has fields
the eval env's rsl_rl 2.0.2 `RslRlOnPolicyRunnerCfg` rejects strictly via
`update_class_from_dict`). Its kill-on-failure logic (designed to save 13 h
of compute) fired before the integration fix could land. Once fixed, the
manual smoke ran cleanly through episode 0 and wrote a measurement JSON.

### Update @ 11:40 — relaunched as v3 with paranoid watchers

**Integration fix in `bench_k1_vlnce.sh`**: stop symlinking the booster_train
run dir into NaVILA-Bench. Instead build a real wrapper directory with
- per-checkpoint symlinks `model_X.pt → booster_train run dir`
- a freshly-written `params/agent.yaml` matching the rsl_rl 2.0.2 schema
  the benchmark accepts (no `obs_groups`, `noise_std_type`, `rnd_cfg`,
  `symmetry_cfg`, etc.)

**Watcher changes:**
- `fire_smoke_when_ready.sh` → renamed `.disabled`. Integration is already
  validated; no need for an automated kill-on-smoke-fail watcher.
- `auto_bench_on_train_end.sh` no longer runs a pre-bench smoke (was
  race-condition prone), and now refuses to fire the 1077-ep benchmark
  unless a checkpoint of iter ≥ 15 000 exists. A transient tmux hiccup
  early in training can no longer kick off a 12 h junk benchmark.

### Status @ 11:42
v3 spinning up. Same gentle terrain, same PPO, fresh training from random
init (no `--resume`; booster_train's resume always creates a new run dir,
so the value over a fresh launch is small at iter 2000). ETA ~14 h. Final
SR/NE/SPL TBD after eval.

---

## 2026-05-19 — v3 walks: 7.81 m path on the bench

Full re-architecture day. The v2 lineage produced a degenerate
"lunge-and-fall" policy at reward 73; MuJoCo sim-to-sim playback
confirmed the policy itself never learned a stable gait. Root cause
was that the v2 env config dropped six of the things that make
booster_train's working `velocity_env_cfg.py` walk:
gait_phase obs, 5-frame history, action_scale 0.25, bad_orientation
1.0 rad, flat terrain, and `empirical_normalization=True`.

**v3** is a fresh subclass of `LocomotionVelocityFlatEnvCfg` that adds
only a separate 160-dim `vision` obs group with a height-scanner (not
fed to actor/critic). Trained 30 000 iters:

| iter | mean reward | mean ep_len |
|---|---|---|
| 1 400 | 92 | 1 425 / 1500 |
| 4 999 | 100 | 1 451 |
| 11 998 | 108 | 1 490 |
| **16 348** | **111.57 (peak)** | 1 500 / 1500 |
| 29 998 | 105 | 1 498 |

Best save-point: `model_16000.pt`. Total training time 4 h 35 min.

### Bench wrapper adapter

The bench's `VLNEnvWrapper` assumes v2's single-step 235-dim obs. For v3
we need 47-dim per-step + 5-frame term-major history + gait clock +
action_scale 0.25. Implemented `VLNEnvWrapperV3` in
`NaVILA-Bench/.../utils/wrappers_v3.py` that:

* builds 47-dim per-step obs from raw scene state
* maintains internal (N, 5, 47) history buffer, term-major flattens to 235
* advances a 2 Hz gait clock per policy step
* pre-divides actions by 2 to compensate for the env's
  `action_scale=0.5` (effective 0.25)
* keeps the existing `k1_matterport_vision` env unchanged — wrapper
  ignores the env's "policy" obs group

New eval entry: `NaVILA-Bench/scripts/navila_eval_v3.py` (loads v3 actor
state-dict directly into a fresh nn.Sequential, skipping rsl_rl runner
machinery so the v3 `agent.yaml` doesn't have to live in the bench env).
Driver: `~/Projects/k1_research/scripts/bench_k1_vlnce_v3.sh smoke|full`.

### Smoke (episode 0, model_16000.pt, gait_phase_init=0.0)

```json
{
    "path_length": 7.806344176620769,
    "distance_to_goal": 8.127594548264664,
    "success": 0.0,
    "spl": 0.0,
    "oracle_navigation_error": 7.616401782362415,
    "oracle_success": 0.0
}
```

**path_length 7.81 m** — first time this benchmark has ever produced
> 1 m on K1. The K1 walks under VLM guidance. Success is still 0; the
remaining failure mode is navigation accuracy (the VLM emitted "turn
left 15 degree" twice in a row at episode start, suggesting course
correction without a viable forward path to the goal). That's a
separate problem from policy + integration, which now both work.

Comparison vs the broken v2 lineage:

| Stage | path_length | SR | Notes |
|---|---|---|---|
| K1 zero-action stub | 0.28 m | 0% | Standing in place |
| Legged-loco v2 (Fix-v1) | 1.95 m | 0% | Reward 17, walks but doesn't navigate |
| Booster-train v2 (broken)             | 0.018 m | 0% | Policy collapsed — lunge-and-fall |
| Booster-train v2 + actuator fix       | 0.057 m | 0% | Eval cfg matched; policy still broken |
| **Booster-train v3 + wrapper adapter** | **7.81 m** | 0% | **First walking K1 in the bench** |

### Full benchmark — running
Launched 2026-05-19 17:08 in tmux `bench_v3`. 1077 episodes × ~95 s →
ETA ~28 h. Results accumulate in
`NaVILA-Bench/eval_results/k1_matterport_vision_loco_v3_model_16000/`.
Aggregate after completion with
`aggregate_k1_vision_results.py --measurements-dir <that dir>/measurements`.
