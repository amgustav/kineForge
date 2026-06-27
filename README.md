# kineForge

**kineForge trains a small MuJoCo robot arm with PPO, stress-tests the learned policy under configurable failures and gate profiles, and writes reproducible local evaluation artifacts.**

It is an RL-first robot policy testbed for local simulation experiments: train, evaluate, compare, sweep configs, and inspect replay plots without a web app, database, or cloud service.

## Quickstart

```bash
git clone https://github.com/amgustav/kineForge.git
cd kineForge

python3 -m venv .venv
source .venv/bin/activate
pip install -e .

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

See [`RESULTS.md`](RESULTS.md) for a committed v0.6.0 result capsule with reproducible commands and example matrix artifacts.

Useful discovery commands:

```bash
python eval.py --list-gates
python eval_matrix.py --list-presets
python sweep.py --list-presets
```

## What it produces

All outputs are local under `runs/`.

- `runs/latest/` — latest policy, scorecard, metadata, config snapshot, and replay plots.
- `runs/train-*/` — timestamped training policy and training metadata.
- `runs/eval-*/` — one policy evaluation with `scorecard.json`, metadata, config snapshot, and PNG plots.
- `runs/eval-matrix-*/` — scenario scorecards plus `matrix_summary.json`, `summary.csv`, `report.html`, and replay index.
- `runs/sweep-*/` — one trained/evaluated policy per variant plus `sweep_summary.json`, `summary.csv`, and `sweep_report.html`.

Scorecards report success rate, final distance, timeout rate, reward, gate status, failed gate criteria, and the active failure modes.

## How it works

- Simulator: MuJoCo through Python bindings.
- Environment API: Gymnasium.
- RL algorithm: Stable-Baselines3 PPO.
- Robot/task: one 2-DoF Reacher-style arm on a tabletop reach-to-target task.
- Configs: YAML for robot, task, reward, failures, gate profiles, matrix presets, and sweep presets.
- Evaluation: deterministic policy rollout, JSON scorecard, static HTML/CSV reports, and matplotlib replay plots.

The default eval matrix runs baseline, target shift, observation noise, action noise, weak actuator combinations, and documented friction placeholders. Gate profiles live in `configs/gates/`; `standard` is the default, `strict` is tighter, and `smoke` is permissive for artifact checks.

## Current limitations

- One robot and one tabletop reaching task.
- Collision/contact metrics are still placeholders.
- Friction failure modes are documented but not physically modeled yet.
- PNG replay only; no video output yet.
- No real robot deployment, web UI, database, cloud service, or broad benchmark suite.

## Roadmap

Next planned work:

1. real contact/collision metrics;
2. better replay and visual inspection;
3. broader task variants only after evaluation quality improves;
4. local experiment indexing and reproducibility polish.

For contributors and coding agents, see `PROJECT_DOCTRINE.md`, `ROADMAP.md`, and `AGENTS.md`.

## Credits / License

Built with MuJoCo, Gymnasium, Stable-Baselines3, NumPy, PyYAML, matplotlib, and pytest.

MIT license.
