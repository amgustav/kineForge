# kineForge

**kineForge trains a small MuJoCo robot arm with PPO, evaluates the learned policy under configurable failures and gates, and writes reproducible local artifacts.**

It is a local-first robot policy evaluation testbed: one tabletop reaching task, YAML configs, deterministic scorecards, static reports, config sweeps, and replay plots under `runs/`.

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

Run the default config sweep:

```bash
python sweep.py --preset default --timesteps 1000 --seed 1
```

Inspect available gates and presets:

```bash
python eval.py --list-gates
python eval_matrix.py --list-presets
python sweep.py --list-presets
```

## Results

[`RESULTS.md`](RESULTS.md) documents the committed v0.6.0 results capsule: exact reproduction commands, seed, timesteps, matrix preset, gate profile, summarized metrics, and what the result proves.

Example artifacts live in [`examples/results/v0.6.0/`](examples/results/v0.6.0/):

- [`matrix_summary.json`](examples/results/v0.6.0/matrix_summary.json) — aggregate and per-scenario metrics.
- [`summary.csv`](examples/results/v0.6.0/summary.csv) — ranked scenario table.
- [`report.html`](examples/results/v0.6.0/report.html) — static matrix report.
- [`replay_gallery.html`](examples/results/v0.6.0/replay_gallery.html) — static trajectory gallery.
- `scenarios/*/scorecard.json` — per-scenario scorecards.
- `scenarios/*/trajectory.png` — replay plots.

Refresh the committed replay gallery locally:

```bash
python replay_gallery.py --summary examples/results/v0.6.0/matrix_summary.json --replay-index examples/results/v0.6.0/replay_index.json --output examples/results/v0.6.0/replay_gallery.html
```

## What it produces

- `runs/latest/` — latest policy, scorecard, metadata, config snapshot, and replay plots.
- `runs/train-*/` — timestamped policy and training metadata.
- `runs/eval-*/` — scorecard, eval metadata, config snapshot, trajectory plot, distance plot, and reward plot.
- `runs/eval-matrix-*/` — scenario scorecards plus `matrix_summary.json`, `summary.csv`, `report.html`, and `replay_index.json`.
- `runs/sweep-*/` — trained/evaluated policy per variant plus `sweep_summary.json`, `summary.csv`, and `sweep_report.html`.
- `runs/run_index.json` and `runs/run_index.csv` — local index of train, eval, matrix, and sweep artifacts.

## How it works

The core loop is:

```text
robot config -> task config -> reward config -> PPO train -> failure eval -> gate -> scorecard -> replay/report
```

- Simulator: MuJoCo through Python bindings.
- Environment API: Gymnasium.
- RL algorithm: Stable-Baselines3 PPO.
- Robot/task: one 2-DoF Reacher-style arm on one tabletop reach-to-target task.
- Configs: YAML for robot, task, reward, failures, gates, matrix presets, and sweep presets.
- Reports: JSON scorecards, JSON summaries, CSV tables, static HTML, and matplotlib PNG plots.

Scope: kineForge is a local simulation/evaluation harness; it does not claim real robot deployment, broad benchmark coverage, or measured contact safety for the current kinematic task.

## Project docs

For contributor and agent workflow details, see PROJECT_DOCTRINE.md, ROADMAP.md, and AGENTS.md.

## Credits / License

Built with MuJoCo, Gymnasium, Stable-Baselines3, NumPy, PyYAML, matplotlib, and pytest.

MIT license.
