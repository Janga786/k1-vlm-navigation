# Quick resume — VLN-CE-Isaac K1 setup

State at last shutdown: pipeline fully operational, 20-episode stub-policy
benchmark complete (all SR=0% as expected with placeholder policy).

## Where everything lives
- **Report:** [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)
- **Setup log (every command + every fix):** [SETUP_LOG.md](SETUP_LOG.md)
- **NaVILA-Bench branch:** `vlnce-isaac-benchmark` (local-only, 3 commits)
- **Conda env for Isaac Sim:** `vlnce-isaac` (Python 3.10)
- **Conda env for NaVILA VLM:** `navila` (already has llava+transformers)
- **Isaac Lab fork:** `~/IsaacLab-vlnce` (patched, do NOT git-pull blindly)
- **NaVILA-Bench:** `~/Projects/k1_research/NaVILA-Bench`
- **Matterport USDs:** `~/Projects/k1_research/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/assets/matterport_usd/` (5.2 GB, 91 scenes)
- **R2R episodes:** `~/Projects/k1_research/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/assets/vln_ce_isaac_v1.json.gz` (1077 episodes)
- **NaVILA checkpoint:** `~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f` (16 GB, llama3-8b-8f)

## Resume the Claude Code conversation
```bash
claude --continue
# or
claude --resume       # pick from list
```

## Rebuild the runtime state (two terminals)

### Terminal 1 — VLM bridge (in the `navila` env)
```bash
conda activate navila
python ~/Projects/k1_research/NaVILA-Bench/scripts/vlm_server_bridge.py \
    --model_path ~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f \
    --port 54321
# wait for: [bridge] listening on localhost:54321
```

### Terminal 2 — single K1 episode test (in the `vlnce-isaac` env)
```bash
conda activate vlnce-isaac
export OMNI_KIT_ACCEPT_EULA=yes
cd ~/Projects/k1_research/NaVILA-Bench
python scripts/navila_eval.py \
    --task=k1_matterport_base --num_envs=1 \
    --load_run=stub_zero_2026-05-15 \
    --headless --enable_cameras --episode_idx=0
```

### Terminal 2 — full benchmark slice
```bash
conda activate vlnce-isaac
export OMNI_KIT_ACCEPT_EULA=yes
cd ~/Projects/k1_research/NaVILA-Bench
python scripts/run_benchmark.py \
    --task=k1_matterport_base \
    --low_level_policy_dir=stub_zero_2026-05-15 \
    --start-idx=0 --end-idx=1077 \
    --per-episode-timeout=240
```

### Re-aggregate measurements
```bash
python - <<'PY'
import json, glob, os, statistics
mdir = os.path.expanduser("~/Projects/k1_research/NaVILA-Bench/eval_results/k1_matterport_base_loco_stub_zero_2026-05-15/measurements")
files = sorted(glob.glob(f"{mdir}/*.json"))
keys = ["success","spl","distance_to_goal","oracle_success","oracle_navigation_error","path_length"]
acc = {k: [] for k in keys}
for f in files:
    d = json.load(open(f))
    for k in keys: acc[k].append(d[k])
print(f"n={len(files)}")
for k in keys:
    print(f"  {k}_mean = {statistics.mean(acc[k]):.4f}")
PY
```

## What's left to do for real numbers
SR will stay 0.0% until you train a real K1 walking policy in the
benchmark's observation layout. See BENCHMARK_REPORT.md
§ "How the placeholder ports to real K1 SR numbers".

Two paths:
- **(A) Train fresh in IsaacLab 1.1 (lower-risk, ~6-12 GPU hours on the 5090)** — mirror the H1 base config's 45-dim proprio obs, use H1's PPO hyperparameters, drop the resulting checkpoint into `logs/rsl_rl/k1_base/<run>/` and re-run `scripts/run_benchmark.py`.
- **(B) Port history-aware obs from Isaac Lab 2.x to 1.1** so the existing `booster_train/logs/rsl_rl/k1_velocity/.../model_9999.pt` works — riskier, but no GPU training needed.

## Clean shutdown before reboot
```bash
# Stop the VLM bridge if it's still running
kill $(pgrep -f vlm_server_bridge.py)
# Anything stuck in Isaac Sim
pkill -9 -f "navila_eval|run_benchmark|demo_planner"
```
