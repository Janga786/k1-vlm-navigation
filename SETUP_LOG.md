# VLN-CE-Isaac Setup Log

Started: 2026-05-15
Host: media (RTX 5090, 32 GB VRAM)
User: janga@media

## Pre-survey (from prior turn)
- Isaac Lab 2.3.0 + Isaac Sim 5.1.0 already installed (Python 3.11, conda env `isaacLab_311`)
- Trained K1 velocity policy: `~/Projects/k1_research/booster/booster_train/logs/rsl_rl/k1_velocity/2026-05-08_14-15-21/model_9999.pt`
- NaVILA checkpoint: `~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f` (16 GB)
- NaVILA-Bench + legged-loco cloned at `~/Projects/k1_research/`
- R2R episodes downloaded: 1077 episodes, 11 unique Matterport scenes

## Decisions made up front
- Per user's "lowest-risk" pick: maintainers' Isaac Lab 1.1 fork in a separate `vlnce-isaac` Python 3.10 conda env.
- RTX 5090 has compute capability 12.0 (Blackwell). Isaac Sim 4.1.0 + PyTorch 2.2.2 cu121 binaries support up to sm_90. **High likelihood of GPU incompatibility**; will document if hit.


## 2026-05-15 — Phase 1 start

### Step 1: conda env
`conda create -n vlnce-isaac python=3.10 -y` — OK.

### Step 2: clone IsaacLab fork
`git clone https://github.com/yang-zj1026/IsaacLab.git ~/IsaacLab-vlnce` — OK, VERSION=1.1.0.

### Step 3: install Isaac Sim 4.1.0
Per README:
```
pip install isaacsim-rl==4.1.0 isaacsim-replicator==4.1.0 \
  isaacsim-extscache-physics==4.1.0 isaacsim-extscache-kit-sdk==4.1.0 \
  isaacsim-extscache-kit==4.1.0 isaacsim-app==4.1.0 \
  --extra-index-url https://pypi.nvidia.com
```
Resolved dependencies pulled **torch 2.12.0+cu130** (not the README's 2.2.2; pip indexes have moved). CUDA 13 toolkit. Surprisingly, this happens to be compatible with RTX 5090 / Blackwell (sm_120).

Sanity: `torch.cuda.get_device_name(0)` = `NVIDIA GeForce RTX 5090`, capability `(12, 0)`, matmul on GPU works.


### Step 4: IsaacLab installer + torch fix
- `./isaaclab.sh -i none` downgraded torch to **2.2.2+cu121** (pinned in extensions). PyTorch 2.2 has NO sm_120 kernels → matmul on 5090 fails with `no kernel image is available for execution on the device`.
- Fix: `pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- Result: **torch 2.11.0+cu128**, matmul on 5090 works.
- Dependency conflict reported (Isaac Lab pins torch==2.2.2) — IGNORED, since 2.11 is functionally compatible and 2.2 doesn't work on Blackwell.

### Step 5: EULA + Isaac Sim smoke
- Non-interactive shell can't accept EULA. Persisted via:
  `echo yes > ~/miniconda3/envs/vlnce-isaac/lib/python3.10/site-packages/omni/EULA_ACCEPTED`
- Isaac Sim 4.1.0 launches headless on RTX 5090, prints `ISAAC_SIM_LAUNCHED_OK` and shuts down cleanly.

### Step 6: Symlink benchmark extensions
- `~/IsaacLab-vlnce/source/extensions/omni.isaac.vlnce` → `~/Projects/k1_research/NaVILA-Bench/.../omni.isaac.vlnce`
- `~/IsaacLab-vlnce/source/extensions/omni.isaac.matterport` → same for matterport
- Installed extension deps: `trimesh PyQt5 pandas`
- Installed `scripts/rsl_rl` (rsl_rl 2.0.2 with onnx + GitPython).


### Step 7: Patch obs/camera bugs for the new stack
Three IsaacLab 1.1.0 + IsaacSim 4.1.0 + RTX 5090 issues found and fixed:

1. **ObservationManager probes camera shape before camera is initialised.**
   `isaac_camera_data`/`process_depth_image` in
   `omni.isaac.vlnce/.../mdp/observations.py` crashed with
   `'Camera' object has no attribute '_is_outdated'` during env __init__.
   Patched: return a zero placeholder of the correct shape when the
   sensor hasn't been initialised yet (sensor.cfg.{height,width}).

2. **Camera.reset() fails because PLAY event hasn't fired in time.**
   `omni.isaac.lab/.../sensors/camera/camera.py:reset` raised
   "Camera could not be initialized. Please ensure --enable_cameras...".
   Patched: try `self._initialize_impl(); self._is_initialized = True`
   first; only raise if that throws.

3. **Replicator pulls boto3/s3transfer; Isaac Kit bundles an older botocore.**
   `ImportError: cannot import name 'DEFAULT_CHECKSUM_ALGORITHM' from
   'botocore.httpchecksum'`. Fix: downgrade
   `s3transfer<0.11 botocore>=1.34,<1.35 boto3<1.35`.

### Phase 1 GO — Go2 baseline
```
python scripts/demo_planner.py --task=go2_matterport_vision \
  --history_length=9 --load_run=2024-09-25_23-22-02 \
  --headless --enable_cameras --episode_index=0
```
Runs end-to-end. PD planner streams `vel_command` tensors to the loaded
Go2 vision policy; sim runs ~56 s of wall-clock before the planner
completes its trajectory. Hydra mesh primvar warnings present but
non-fatal (cosmetic, Go2 USD only). **Phase 1 GO.**


## Phase 2 — K1 robot config (DONE pipeline-wise)

Created `isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/config/k1/`:
- `__init__.py` registers `k1_matterport_base` gym id.
- `k1_matterport_base_cfg.py` — H1-style task config but with K1
  locomotion URDF, head camera on Trunk +0.85m, 12-DoF leg action,
  proprio-only observation (no height scan), no contact-sensor terms
  (Isaac Lab 1.1 URDF importer doesn't expose `activate_contact_sensors`).

Stub low-level policy:
- `scripts/.../build_k1_stub_v2.py` (one-shot script) writes
  `logs/rsl_rl/k1_base/stub_zero_2026-05-15/{model_0.pt,params/{agent,env}.yaml}`
- Actor MLP last layer zeroed, log-std = -5 → deterministic zero action.
- The booster_train trained K1 policy (`k1_velocity/.../model_9999.pt`)
  uses `history_length=5, flatten_history_dim=True` observations which
  Isaac Lab 1.1's ObservationGroupCfg does not support — REUSE IS NOT
  POSSIBLE without porting Isaac Lab 2.x's history mechanism backwards.
  Stub is the pragmatic choice; the robot doesn't walk, but the
  benchmark's data flow runs.

### Phase 2 GO — K1 demo_planner runs:
```
python scripts/demo_planner.py --task=k1_matterport_base \
  --load_run=stub_zero_2026-05-15 --headless --enable_cameras \
  --episode_index=0
```
URDF import (slow first time, ~90s to convert K1_locomotion.urdf to USD),
scene load, vel_command stream from the PD planner. The stub policy
keeps the K1 at default joint pose so it tips over within ~1 s of sim
time, but the eval pipeline plumbing works end-to-end.


## Phase 3 — NaVILA VLA connect

### VLM bridge server
`scripts/vlm_server_bridge.py` (created) — TCP server that speaks the
benchmark's 8-byte-size-prefix JSON protocol and runs NaVILA in the
existing `navila` conda env (torch 2.7.1+cu128, transformers 4.49.0,
llava + NaVILA from `~/Projects/k1_research/booster/NaVILA`).

Why a separate process: the `vlnce-isaac` env has Isaac Sim 4.1
which doesn't coexist with llava + flash-attn cleanly. The `navila`
env already loads the model. Two-process split keeps both stacks
isolated.

Started with:
```
conda activate navila
python scripts/vlm_server_bridge.py \
  --model_path ~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f \
  --port 54321
```
Logs `[bridge] listening on localhost:54321` once warm.

### navila_eval Z-offset for K1
Patched `scripts/navila_eval.py:131-138` to add an explicit `k1`
branch (z += 0.55).

### Single-episode run command
```
conda activate vlnce-isaac
OMNI_KIT_ACCEPT_EULA=yes python scripts/navila_eval.py \
  --task=k1_matterport_base --num_envs=1 \
  --load_run=stub_zero_2026-05-15 \
  --headless --enable_cameras --episode_idx=0
```

### Phase 3 fixes (additional)
- `vlnce/vlnce/mdp/actions/navigation_actions.py`, `vlm_navigation_actions.py`, `vlm_navigation_actions_gpt.py`: removed module-top `import torchvision` (transitively imported PIL from Isaac Sim's pip_archive — version mismatch crash on `_util.is_directory`). The references were all in commented-out resize_transform blocks.
- Downgraded Pillow in the vlnce-isaac env to **10.2.0** to match Isaac Sim's bundled Pillow (fixes JPEG-encoder ABI mismatch: `_getencoder takes at most 16 arguments (17 given)`).
- Patched `scripts/navila_eval.py:131-138` to add a `k1` z-offset branch (+0.55).

### Phase 3 GO — full pipeline works
End-to-end with NaVILA driving K1 in a Matterport scene:
- Bridge VLM in `navila` env (port 54321) returned commands like
  `'The next action is turn left 15 degree.'` in ~330-680 ms.
- `scripts/navila_eval.py` consumed those, computed velocity commands
  via `get_vel_command()`, stepped K1 in the Matterport USD scene.
- Episode 0 measurement written to
  `eval_results/k1_matterport_base_loco_stub_zero_2026-05-15/measurements/0.json`:
  ```
  path_length: 0.056 m, distance_to_goal: 8.37 m,
  success: 0.0, spl: 0.0, oracle_success: 0.0
  ```
- SR=0 is the EXPECTED outcome with the zero-action stub — the K1
  doesn't walk, just receives commands and tips over. The point is the
  data flow runs end-to-end; SR > 0 requires training a real K1 policy
  in the benchmark's 45-dim proprio + 12-dim action layout.


## Phase 4 — Multi-episode benchmark
Running `python scripts/run_benchmark.py --task=k1_matterport_base --low_level_policy_dir=stub_zero_2026-05-15 --start-idx=1 --end-idx=20 --per-episode-timeout=240` in background; each episode hits a hard 4-min timeout (Isaac Sim cold-starts each call, ~50s, then sim runs until VLM emits "stop" or max-steps). Expected total: ~30-40 min for 19 episodes. Per-episode JSON drops in `eval_results/.../measurements/<i>.json`.

## Phase 5 — Report written
`BENCHMARK_REPORT.md` produced. Includes:
- headline metric table with K1 vs paper Go2/H1
- commands that work for future re-runs
- patch inventory across IsaacLab-vlnce, NaVILA-Bench, and the conda env
- "What's broken" with the stub-policy caveat
- Two paths to real K1 SR numbers (retrain in benchmark vs port history obs backwards)
- One-shot aggregator script in the bottom of the report

## Branch state
- New branch: `vlnce-isaac-benchmark` on NaVILA-Bench fork
- Two commits: K1 config + scaffolding/fixes
- NOT pushed to remote (no remote configured for NaVILA-Bench)


## Phase 4 — DONE
19/19 episodes ran. Final aggregate:
- success_mean = 0.0000 (stub policy never walks)
- distance_to_goal_mean = 10.86 m
- path_length_mean = 0.29 m

All 19 measurement JSONs + MP4 videos under
`eval_results/k1_matterport_base_loco_stub_zero_2026-05-15/`.

## Done
All 5 phases complete. Pipeline proven end-to-end. SR=0% is the
expected stub-policy outcome; replace the stub with a trained K1
walking policy (in 45-dim proprio + 12-dim leg-action layout) and
re-run `scripts/run_benchmark.py` to get meaningful numbers.

---

## 2026-05-15 23:17 — Overnight: K1 vision training

### Step 1: Install tmux
- tmux missing → `conda install -y -n base -c conda-forge tmux` (v3.6a).

### Step 2: Read legged-loco config structure
- legged-loco has go1, go2, h1, g1 configs (humanoid + quadruped).
- H1 vision config (`h1_low_vision_cfg.py`) is the closest template (humanoid + RayCaster).
- legged-loco uses Isaac Lab 1.1 namespace (`omni.isaac.lab.*`), same as vlnce-isaac env.

### Step 3: Extract K1 physics
- From `booster_train/source/.../assets/robots/booster.py`:
  - K1_locomotion URDF: 12 leg DoF (Hip Pitch/Roll/Yaw, Knee Pitch, Ankle Pitch/Roll × 2).
  - PD: stiffness=350 hips/knees, 250 ankles; damping=7.5/5.
  - Standing height: z=0.55m.
  - Default joints: Hip_Pitch=-0.15, Knee_Pitch=0.30, Ankle_Pitch=-0.15, others=0.
  - Body names after merge_fixed_joints: Trunk, left_foot_link, right_foot_link, Left_Shank, etc.

### Step 4: Write K1 vision config
- `legged-loco/.../config/k1/k1_low_vision_cfg.py` (new, ~430 LOC).
  - URDF loaded via `UrdfFileCfg` with `merge_fixed_joints=True`, `activate_contact_sensors=True`
    (contradicts the k1_matterport_base note — Isaac Lab 1.1 DOES support contact sensors
    on URDF-spawned articulations).
  - Rough terrain: stairs, boxes, slopes, init-pos obstacles (height range 0.05-0.20m,
    smaller than H1's 0.30 to suit the 0.55m-tall K1).
  - K1Rewards inherits RewardsCfg + termination_penalty -200, track_lin/ang_vel exp,
    feet_air_time_positive_biped on `.*foot_link`, feet_slide, dof_pos_limits on
    `.*_Ankle_(Pitch|Roll)`, joint_deviation_l1 on `.*_Hip_(Yaw|Roll)`.
  - Skipped H1's joint_deviation_arms / _torso (K1 locomotion URDF has no arm/torso joints
    after merge).
  - K1RoughPPORunnerCfg: experiment_name=k1_vision_rough, max_iter=2000, save_interval=200.
- `legged-loco/.../config/k1/k1_low_base_cfg.py` (new) — blind fallback (no height_scan).
- `legged-loco/.../config/k1/__init__.py` (new) — gym.register k1_base, k1_base_play,
  k1_vision, k1_vision_play.
- `legged-loco/.../config/__init__.py` — added `from .k1 import *`.

### Step 5: Install + launch
- `pip install -e legged-loco/isaaclab_exts/omni.isaac.leggedloco --no-deps`.
- Launched in tmux session `k1train` (Isaac Sim 4.1 + IsaacLab 1.1, vlnce-isaac env):
  ```
  python scripts/train.py --task=k1_vision --run_name=k1_vision_v1 \
      --max_iterations=2000 --save_interval=200 --headless
  ```
- Confirmed at iter ~20: env builds (4096 envs), 12 reward terms active, 2 terminations,
  contact_forces sensor working (feet_air_time, feet_slide, base_contact all populating).
- Iter time: 6.26s. ETA: ~3 hours.
- Early reward signals (iter 19):
  - track_lin_vel_xy_exp 0.011 (still tiny — expected pre-policy)
  - feet_air_time 0.0001 (positive, healthy)
  - termination_penalty -0.20 (high — robot falls often, expected baseline)
  - base_contact 117/iter (every env terminating from base hitting ground, expected at iter 19)

### Step 6: Prepare benchmark integration (in parallel with training)
- `NaVILA-Bench/.../config/k1/k1_matterport_vision_cfg.py` (new, ~290 LOC).
  - Same `K1_ARTICULATION_CFG` as training config.
  - Observation order matches `k1_low_vision_cfg.PolicyCfg` exactly (base_lin/ang_vel,
    gravity, vel cmd, joint_pos/vel, last_action, height_scan).
  - Height scanner attached to Trunk (matches training).
  - RGB camera mounted on Trunk at (0.10, 0, 0.85) (K1 head height).
- Registered `k1_matterport_vision` gym task in `config/k1/__init__.py`.
- Wrote `run_k1_benchmark.sh` end-to-end pipeline script: symlinks trained policy from
  legged-loco→NaVILA-Bench, starts VLM bridge in tmux, runs single-episode smoke test,
  then launches full benchmark in tmux `bench`.

### Decisions
1. **Vision config** (not blind) — per user requirement (NaVILA Table IV vision row).
2. **Action scale 0.5** — matches H1 vision config; booster_train K1 used 0.25 but that's
   for the 47-dim/5-frame-history obs, so 0.5 better matches H1's 1-frame policy network.
3. **Heading command** — kept H1's `heading_command=True` for training stability; benchmark
   eval uses direct ang_vel_z from NaVILA.
4. **Step heights 0.05-0.20m** — half the H1 range (0.05-0.30) since K1 is ~half H1's height.
5. **No domain-random base mass** for now — disabled in `__post_init__` (`add_base_mass=None`)
   to make early reward learning easier; can re-enable mid-training if reward plateaus.

### Step 7: Overnight extension — v1 done, v2 resumed, benchmark launched

After v1 finished at iter 1999 (01:11, ~1h54), user extended training. The
in-tmux watcher chain had a critical bug — at some point during the v1→v2
handoff, the conda `tmux` binary symlink got replaced by a socket file
(it's `~/miniconda3/bin/tmux` → suddenly `srw-------`). Cause is unclear; possibly
some interaction between `tmux new-session` inside a tmux session at the
exact moment the running `tmux` binary path gets touched.

Mitigation: switched to a single nohup+setsid `mega_pipeline.sh` that does
training → benchmark sequentially. No tmux required. The mega_pipeline ran:
- v2 with `--resume True --load_run=<v1_dir> --max_iterations=8000`
  (total iters = 2000 v1 + 8000 = 10000, exactly what user wanted)
- `timeout(1) 21600` for 6h wall-clock cap on v2 so benchmark has time.

v2 hit cap at 07:16:13 (exit code 124) at iter 9000. Mean reward grew from
v1's ~8.4 → v2's ~17 (2× improvement). The watcher chose v2 (iter 9000 >
v1's iter 1999) and auto-launched the benchmark at 07:16:43.

Benchmark live at 07:56: 64 episodes, SR=0%, path_length 0.70 m (vs stub
0.28 m). K1 walks but doesn't reach goals 7-13 m away. Likely needs
substantially more training (paper's H1 vision used ~50k iters).


## 2026-05-18 — Phase 3: pivot to booster_train

### Step 8: legged-loco v3 (50 k iter) collapsed

v3 ran 2026-05-16 12:47 → 2026-05-18 06:38 (~42 h). Finished iter 49,999.
**Diverged at iter ~44,910** — `action_rate_l2` exploded from -0.04 to -3.6e12,
then to -3.085e+15 by the end. Mean reward dropped to -12 to -21.

Root cause: legged-loco's `K1.rewards.action_rate_l2.weight = -0.005`
(`legged-loco/.../k1_low_vision_cfg.py:619`) vs booster_train's `-1.0`
(`booster_train/.../velocity_env_cfg.py:273`) — 200× weaker penalty +
unclipped post-tanh actions = late-stage policy oscillation, then divergence.
`dof_torques_l2 = 0` and `dof_acc_l2 ≈ -0.09` at the same iter, confirming the
robot itself wasn't moving violently — only the policy *outputs* were
oscillating between extremes.

### Step 9: design new training task — `Booster-K1-Velocity-Vision-VLNCE-v0`

Per user direction: train a fresh policy in **booster_train** (Isaac Lab 2.3 /
Isaac Sim 5.0, conda env `isaacLab_311`) that drops into the
`vlnce-isaac` benchmark with **zero adapter**.

Files added (new only, no edits to existing booster_train tasks):
- `booster_train/source/.../velocity/robots/k1/vlnce_env_cfg.py`
- `booster_train/source/.../velocity/robots/k1/vlnce_ppo_cfg.py`
- `booster_train/source/.../velocity/robots/k1/__init__.py` — appended
  `gym.register` for `Booster-K1-Velocity-Vision-VLNCE-v0` and `-Play`.

Design (benchmark-compat constraints):
1. **Policy obs = 235 dim**, exact match to benchmark `k1_matterport_vision_cfg.py`
   PolicyCfg order:
   `base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
    + joint_pos_rel(12) + joint_vel_rel(12) + last_action(12) + height_scan(187)`.
   The 187 (not 160 as initially estimated) comes from
   `GridPatternCfg(resolution=0.1, size=(1.6, 1.0))` → `17×11` rays.
2. **Critic obs = same 235** (same terms, no noise).
3. **`history_length=0`** (benchmark doesn't stack), **no `gait_phase`**
   (benchmark doesn't have it), **`empirical_normalization=False`**
   (benchmark agent config uses Identity).
4. **Action scale 0.5** (benchmark default), `use_default_offset=True`.
5. **PD gains from `BOOSTER_K1_LOCOMOTION_CFG`** (hip/knee 350/7.5, ankle 250/5)
   — small damping mismatch vs benchmark (350/10, 250/5); policy should
   generalize given domain randomization.
6. **Terminations**: `time_out`, `root_height_below_minimum @ 0.30 m`,
   `bad_orientation @ 1.3 rad` (matches benchmark fix_v1).
7. **Rewards**: full Booster-Gym set carried over including
   `action_rate_l2 weight = -1.0` (the diff vs legged-loco that prevents the
   v3 collapse pattern).
8. **Scene additions in `__post_init__`**:
   - `RayCasterCfg("height_scanner")` on `Trunk`, 20 m above, 16×10@0.1 m grid.
   - Terrain switched from flat plane to `ROUGH_TERRAINS_CFG` (Isaac Lab preset:
     stairs±, boxes, rough, slopes±) with `max_init_terrain_level=0` and
     built-in curriculum.
9. **PPO** (unchanged from booster_train K1): 4096 envs, 24 steps/env,
   5 epochs, 4 minibatches, lr 1e-3 adaptive KL, clip 0.2, entropy 0.01,
   γ=0.99, λ=0.95, max_grad_norm 1.0, init_noise_std=1.0,
   actor/critic [512,256,128] elu.

### Step 10: rsl_rl cross-version compat verified

| Env | rsl_rl | torch | Isaac Lab | Isaac Sim |
|---|---|---|---|---|
| `isaacLab_311` (train) | 3.1.2 | 2.7.0+cu128 | 0.50.5 | 5.1.0 |
| `vlnce-isaac` (eval)   | 2.0.2 | 2.9.1+cu128 | 0.20.8 | 4.1.0 |

`ActorCritic.__init__` signature differs (3.1.2 takes TensorDict obs +
obs_groups; 2.0.2 takes `num_actor_obs` flat) — but the **state_dict keys
are identical**:
`actor.{0,2,4,6}.{weight,bias}`, `critic.{0,2,4,6}.{weight,bias}`, `std`.
Verified by inspecting the existing 10k checkpoint and instantiating ActorCritic
2.0.2 with matching dims — keys & shapes line up. So a 3.1.2-trained `.pt`
loads directly via the benchmark's `OnPolicyRunner.load()` → no ONNX, no
key-remap, no `navila_eval.py` edits required.

### Step 11: smoke test PASSED

```bash
cd ~/Projects/k1_research/booster/booster_train
conda activate isaacLab_311
python scripts/rsl_rl/train.py \
    --task Booster-K1-Velocity-Vision-VLNCE-v0 \
    --num_envs 2 --max_iterations 2 --headless
```
2 envs × 24 steps × 2 iters = 96 timesteps. Mean reward -25.6 (random init).
ObservationManager prints `policy[7] = height_scan (187,)` confirming obs space.
Two harmless warnings:
- `[DEPRECATION] Merging bodies with inertia` — Isaac Sim URDF arm/head merge.
- `rsl_rl.utils.utils:245 UserWarning: 'obs_groups' dict missing 'policy' key` —
  3.1.2 auto-detects, behavior to be removed in a future version.

### Step 12: full training launched 10:07:57

```bash
bash ~/Projects/k1_research/scripts/train_k1_vlnce.sh k1_vlnce_v1 4096 30000
```
- tmux session `k1_vlnce_train` (libtinfo fix:
  `LD_LIBRARY_PATH=$HOME/miniconda3/lib` prefix; bare `tmux ls` fails).
- Log: `/tmp/k1_booster_train.log`.
- Run dir: `booster_train/logs/rsl_rl/k1_velocity_vlnce/2026-05-18_10-08-01_k1_vlnce_v1/`.
- Checkpoints every 2000 iters (per `vlnce_ppo_cfg.save_interval=2000`).

Plateau watcher running in nohup (`/tmp/k1_vlnce_watcher.log`): kills the
training tmux if mean reward hasn't improved by > 0.5 in 15k iters, or drops
by > 20 from best. Records best ckpt name to `/tmp/k1_vlnce_best_ckpt.txt`.

Initial: iter 1 reward -0.81; iter 84 reward -0.72; iter time ~1.67 s
(slower than the 10k flat run's ~0.54 s — vision obs + RayCaster + rough
terrain). ETA at this rate: ~13.9 h to iter 30k. Velocity tracking error
`error_vel_xy` falling from 0.0084 → 0.0064 in the first 80 iters.

### Step 13: benchmark integration script ready

`~/Projects/k1_research/scripts/bench_k1_vlnce.sh`:
- Symlinks the booster_train run dir into
  `NaVILA-Bench/logs/rsl_rl/k1_vision_rough/booster_vlnce_<run_subdir>/`
  so the benchmark resolves the checkpoint via its standard
  `--load_run` mechanism.
- Subcommand `smoke <run_subdir>`: 1 episode, foreground, prints metrics.
- Subcommand `full <run_subdir> [start] [end]`: launches in tmux
  `bench_vlnce`, full 1077 episodes, 240 s/ep timeout.
- VLM bridge auto-started if not already running.
- New `eval_results/` dir name encodes the run, so it stays distinct from
  the legged-loco v2/v3 result dirs (which are untouched per user direction).

Will trigger smoke test once `model_2000.pt` lands (~iter 2000 / 1.67 s = ~56 min
into training). If smoke passes, full run waits until reward has plateaued
or watcher signals best ckpt.

### Step 14: v1 dead-on-arrival, terrain too aggressive — pivoted to v2

By iter 1000, v1 was stuck: **`Mean episode length: 1.00`** (literally 1 step
per env per episode), `Episode_Termination/bad_orientation: 0.7984`, and
`Episode_Reward/track_lin_vel_xy: 0.0022` flat. K1 was tipping past 1.3 rad
within the first physics step of every reset.

Diagnosis: cross-referenced the working flat-ground baseline tfevents
(`booster_train/logs/.../2026-05-08_14-15-21/events.out.tfevents.*`):

| Metric | Flat baseline @ iter 100 | v1 (rough) @ iter 100 |
|---|---|---|
| Mean reward | -2.08 | -0.71 |
| Mean episode length | 13.4 | **1.00** |
| bad_orientation termination | 99.79% | 79.59% |
| track_lin_vel_xy | 0.0075 | 0.0022 |

Flat baseline survived 14 steps at iter 0, then broke out at iter ~500 to
1293 steps (final 1490). v1 never got past 1 step.

Root cause: `ROUGH_TERRAINS_CFG` includes `pyramid_stairs` and `boxes`
sub-terrains with step heights up to **0.23 m** and box heights up to **0.20 m**.
For K1 (0.55-m frame, ~0.25-m max ankle lift) that's a hard problem. With
`pose_range x/y ±0.5 m`, even at curriculum row 0 a random spawn often
landed K1 with one foot on a step and the other off → instant tilt past
1.3 rad → bad_orientation termination on the very first physics step,
before the policy got to act at all.

### Fix: K1-tailored gentle terrain (`K1_VLNCE_TERRAINS_CFG`)

Replaced the preset with a custom config in `vlnce_env_cfg.py`:
- **60% `HfRandomUniformTerrainCfg`** with `noise_range=(0.01, 0.06)` — gentle
  random bumps, height_scan picks up signal at higher difficulty rows.
- **20% `HfPyramidSlopedTerrainCfg`** with `slope_range=(0.0, 0.15)` (~8.5°) —
  walkable ramps.
- **20% `HfInvertedPyramidSlopedTerrainCfg`** with the same gentle range.
- No stairs, no boxes — those need a dedicated stair-walking policy.
- `curriculum=True`, `num_rows=10`, `max_init_terrain_level=0`.

Smoke test (4 envs × 3 iters): episode length **44.5 steps at iter 2**
(vs v1's 1.00 at all iters). bad_orientation 33% at iter 2, dropping.
Confirms the policy can now actually start learning.

### Step 15: relaunched as v2

```bash
bash ~/Projects/k1_research/scripts/train_k1_vlnce.sh k1_vlnce_v2 4096 30000
```
- run dir: `booster_train/logs/.../2026-05-18_10-42-27_k1_vlnce_v2/`
- iter 17: mean reward -14.77, episode length 41.82, bad_orientation 26.15%
  and dropping, track_lin_vel_xy 0.0157 (7× v1 at same iter).
- ETA 13:37 to iter 30 k.
- Smoke-fire + auto-bench + plateau watchers re-armed for `_k1_vlnce_v2`
  pattern; old v1 watchers cleaned up. Stale `bench_vlnce` tmux session
  (spawned by the old auto_bench fired between v1 kill and pkill catching
  it) was killed to free GPU.

### Step 16: v2 progress — breakout at iter 500

| iter | reward | episode length |
|---|---|---|
| 17 | -14.77 | 41.82 |
| 100 | -1.02 | 19.47 |
| 500 | **+29.52** | **978.81** |
| 1000 | +47.72 | 1422.67 |
| 2000 | +56.34 | 1407.20 |

Same breakout pattern as the working flat-ground baseline (which jumped to
+70 / 1293 at iter 500). Reward growing more slowly than baseline (terrain
is harder), but trajectory is healthy. No action_rate divergence — that
penalty stays bounded with the booster_train weight of -1.0.

### Step 17: rsl_rl 3.1.2 ↔ 2.0.2 agent.yaml schema mismatch

At iter 2000, the smoke watcher fired `bench_k1_vlnce.sh smoke ... 2000`.
The first attempt CRASHED at:

```
File ".../navila_eval.py", line 266, in main
    update_class_from_dict(agent_cfg, log_agent_cfg_dict)
KeyError: '[Config]: Key not found under namespace: /obs_groups.'
```

Root cause: booster_train (rsl_rl 3.1.2) writes `params/agent.yaml` with
several 3.1.2-only fields that don't exist on the eval env's rsl_rl 2.0.2
`RslRlOnPolicyRunnerCfg`:
`obs_groups`, `noise_std_type`, `state_dependent_std`,
`actor_obs_normalization`, `critic_obs_normalization`,
`normalize_advantage_per_mini_batch`, `rnd_cfg`, `symmetry_cfg`,
`clip_actions`, top-level `class_name`. `update_class_from_dict` is
strict — any unknown key in the yaml overlay raises `KeyError`.

State-dict load itself is unaffected — the actual ActorCritic weights only
use `actor.{N}.{weight,bias}`, `critic.{N}.{weight,bias}`, `std`, which
match between versions. The break is purely in the dict-overlay before
weight load.

**Fix** (in `bench_k1_vlnce.sh`): stop symlinking the entire booster_train
run dir. Instead, create a real `booster_vlnce_<subdir>/` directory with:
- per-checkpoint symlinks: `model_X.pt -> booster_train/.../model_X.pt`
  for every X present, refreshed each invocation.
- a sanitized `params/agent.yaml` written from scratch, matching the
  legged-loco v2 schema that rsl_rl 2.0.2 accepts (no 3.1.2-only fields).

After the fix, smoke at iter 2000 ran cleanly through episode 0 and wrote:
```json
{ "path_length": 0.0044, "distance_to_goal": 8.40,
  "success": 0.0, "spl": 0.0,
  "oracle_navigation_error": 8.40, "oracle_success": 0.0 }
```
K1 barely moved (iter 2000 policy reward 56, baseline reached 100+ by iter
5000) — expected at this point. Integration is end-to-end working.

### Step 18: v2 killed by its own smoke watcher → v3

The `fire_smoke_when_ready.sh` watcher, by design, fires on `model_2000.pt`
arrival and kills the training tmux session if the smoke fails — so we
don't waste 13 h compute on a doomed integration. At 11:29:24 the watcher
fired the smoke against the (still pre-fix) symlink-only wrapper, hit
the `obs_groups` KeyError, and:

```
[2026-05-18 11:29:31] INTEGRATION FAILURE — no measurement at .../0.json
[2026-05-18 11:29:31] killing training tmux session 'k1_vlnce_train' to save ~13 h of wasted compute.
```

v2 training was at iter 2012, reward 56.34, ep_len 1407 — fine, just
caught in the cross-fire while I was simultaneously fixing
`bench_k1_vlnce.sh`. The kill saved no compute (the integration was
fixable in a few seconds) but also irreversibly stopped the training run.

### Step 19: relaunched as v3 with safer watchers

```bash
bash ~/Projects/k1_research/scripts/train_k1_vlnce.sh k1_vlnce_v3 4096 30000
```
- run dir: `booster_train/logs/.../2026-05-18_11-40-04_k1_vlnce_v3/`
- v3 lost ~50 min of v2 compute (decided not to `--resume` from
  v2's model_2000.pt — booster_train resume always creates a new run dir,
  so the value over a fresh launch is small at this stage).

**Watcher changes (paranoid mode):**
- `fire_smoke_when_ready.sh` → renamed to `.disabled`. Won't fire again.
  Integration already validated manually; no need for a separate watcher
  that can kill training on transient state.
- `auto_bench_on_train_end.sh` simplified:
  - **No pre-bench smoke** — was race-condition-prone (two Isaac Sim
    instances briefly overlapping while training was still wrapping up).
  - **MIN_ITER_FOR_BENCH = 15000** — the watcher refuses to fire the
    1077-ep benchmark unless a checkpoint of iter ≥ 15 k is present, so
    a transient tmux hiccup early in training can't kick off a 12-hour
    junk benchmark.
- `watch_k1_vlnce_train.sh` (plateau watcher) — unchanged. Same plateau
  detection logic.
