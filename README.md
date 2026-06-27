# kineForge

**kineForge trains a small MuJoCo robot arm with PPO, evaluates the learned policy under configurable failures and gates, and writes reproducible local artifacts.**

It is a local-first robot policy evaluation testbed: one tabletop reaching task, YAML configs, deterministic scorecards, static reports, replay plots, and run indexes under `runs/`.

## Quickstart

```bash
git clone https://github.com/amgustav/kineForge.git
cd kineForge
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Train and evaluate a smoke policy:

```bash
python train.py --task tabletop_reach --robot arm_v0 --timesteps 1000 --seed 1
python eval.py --policy runs/latest/policy.zip --seed 1
```

Run the default failure matrix:

```bash
python eval_matrix.py --policy runs/latest/policy.zip --preset default --seed 1
```

Run the default reward/config sweep:

```bash
python sweep.py --preset default --timesteps 1000 --seed 1
```

Useful discovery commands:

```bash
python eval.py --list-gates
python eval_matrix.py --list-presets
python sweep.py --list-presets
python replay_gallery.py --summary examples/results/v0.6.0/matrix_summary.json --replay-index examples/results/v0.6.0/replay_index.json --output examples/results/v0.6.0/replay_gallery.html
python run_index.py --runs-dir runs --output runs/run_index.json --csv runs/run_index.csv
```

## Results

- `RESULTS.md` summarizes the v0.6.0 example outputs and reproduction commands.
- `examples/results/v0.6.0/` contains a matrix summary, CSV table, static report, replay gallery, replay index, per-scenario scorecards, and selected trajectory PNGs.

## What it produces

- trained PPO policies under `runs/latest/` and `runs/train-*/`
- one-policy eval artifacts under `runs/eval-*/`
- matrix reports under `runs/eval-matrix-*/`
- sweep reports under `runs/sweep-*/`
- local run indexes as `runs/run_index.json` and `runs/run_index.csv`

## How it works

- Simulator: MuJoCo through Python bindings
- Environment API: Gymnasium
- RL algorithm: Stable-Baselines3 PPO
- Robot/task: one 2-DoF Reacher-style arm on a tabletop reach-to-target task
- Configs: YAML for robot, task, reward, failures, gates, matrix presets, and sweep presets
- Reports: JSON scorecards, JSON summaries, CSV tables, static HTML, and matplotlib PNG replay plots

Collision/contact metrics are explicit unmeasured placeholders in the current kinematic task. Friction failure modes are documented in scorecards as not physically modeled yet. kineForge does not claim real robot deployment, broad benchmark superiority, or production robotics readiness.

## Files

- `train.py` — train a PPO policy
- `eval.py` — evaluate one policy
- `eval_matrix.py` — evaluate one policy across failure scenarios
- `sweep.py` — train/evaluate config variants
- `compare_eval.py` — compare matrix summaries
- `replay_gallery.py` — build a static replay gallery from matrix artifacts
- `run_index.py` — index local run artifacts
- `kineforge/` — environment, configs, gates, evaluation, reports, sweeps, gallery, and registry helpers
- `configs/` — YAML presets and baseline robot/task/reward/failure/gate configs
- `tests/` — smoke and artifact tests
- `examples/results/v0.6.0/` — curated example result artifacts

For contributor and agent workflow details, see `PROJECT_DOCTRINE.md`, `ROADMAP.md`, and `AGENTS.md`.

## Credits / License

Built with MuJoCo, Gymnasium, Stable-Baselines3, NumPy, PyYAML, matplotlib, and pytest.

MIT license.
