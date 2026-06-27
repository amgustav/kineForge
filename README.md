# kineForge

**Train a small MuJoCo tabletop-reaching policy, evaluate it across failure-mode matrices, and inspect the run through gates, scorecards, reports, replay galleries, and a local artifact index.**

kineForge trains a Stable-Baselines3 PPO policy for a 2-DoF Reacher-style arm on a tabletop reach-to-target task. Evaluation is config-driven: YAML robot/task/reward/failure files feed deterministic rollouts, named gate profiles, eval matrix presets, and reward/task config sweeps. Each run writes inspectable local artifacts under `runs/`: scorecards, metadata, config snapshots, CSV/HTML summaries, replay plots, replay galleries, and JSON/CSV run indexes.

## Result

Current committed example output: `examples/results/v0.6.0/`, generated from a locally trained `arm_v0` / `tabletop_reach` / `reach_v0` policy.

| Field | Value |
| --- | --- |
| Policy source | `runs/latest/policy.zip` from `python train.py --task tabletop_reach --robot arm_v0 --timesteps 25000 --seed 1` |
| Matrix command | `python eval_matrix.py --policy runs/latest/policy.zip --preset default --gate standard --episodes 3 --seed 1` |
| Matrix preset | `default` |
| Gate profile | `standard` |
| Scenarios | 7 scenarios, 3 episodes each |
| Gate result | 7 / 7 PASS, pass rate 1.000 |
| Aggregate success rate | 1.000 |
| Aggregate mean final distance | 0.0451 m |
| Aggregate timeout rate | 0.000 |
| Best ranked scenario | `observation_noise`, PASS, success 1.000, mean final distance 0.0399 m |
| Worst ranked scenario | `target_shift`, PASS, success 1.000, mean final distance 0.0495 m |
| Scenario set | `observation_noise`, `baseline`, `high_friction`, `low_friction`, `action_noise`, `combined_hard`, `target_shift` |
| Artifacts | `matrix_summary.json`, `summary.csv`, `report.html`, `replay_index.json`, `replay_gallery.html`, `scenarios/*/scorecard.json`, `scenarios/*/trajectory.png` |
| Replay gallery | `examples/results/v0.6.0/replay_gallery.html` |

`low_friction` and `high_friction` are documented placeholders in the current kinematic task; contact/collision values are not real safety measurements.

## Quickstart

```bash
git clone https://github.com/amgustav/kineForge.git
cd kineForge
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

python train.py --task tabletop_reach --robot arm_v0 --timesteps 1000 --seed 1
python eval.py --policy runs/latest/policy.zip --seed 1
python eval_matrix.py --policy runs/latest/policy.zip --preset default --seed 1
python sweep.py --preset default --timesteps 1000 --seed 1
python run_index.py --runs-dir runs --output runs/run_index.json --csv runs/run_index.csv
```

## What you get

| Surface | Output |
| --- | --- |
| Training | PPO policy zip, training metadata, config snapshot, timestamped run directory |
| Eval | JSON scorecard, eval metadata, trajectory plot, distance plot, reward plot, latest-run copy |
| Failure matrix | Named scenarios, per-scenario scorecards, ranked `matrix_summary.json`, `summary.csv`, static `report.html` |
| Gate profiles | YAML thresholds, PASS/FAIL status, failed criteria, gate explanation embedded in scorecards and summaries |
| Config sweeps | Named reward/task variants, one policy per variant, ranked `sweep_summary.json`, `summary.csv`, `sweep_report.html` |
| Replay | Matplotlib trajectory PNGs plus static gallery generation from matrix summaries and replay indexes |
| Registry/index | `run_index.py` scans train/eval/matrix/sweep artifacts into JSON and CSV indexes |
| Reports | Machine-readable JSON first; CSV and static HTML for quick inspection without a service |

## How it works

Robot RL experiments often stop at the trained policy file. kineForge keeps the evaluation trail explicit: load YAML configs, train PPO, snapshot the configs, run deterministic evaluation episodes, inject named failure modes, apply gate criteria, write a scorecard, render replay artifacts, and index the resulting run directories.

The simulator is MuJoCo exposed through a Gymnasium environment. Stable-Baselines3 PPO controls a compact Reacher-style arm with joint-delta actions, distance-to-target reward terms, optional observation/action perturbations, and task/reward values loaded from YAML. Matrix and sweep CLIs reuse the same eval path so single-policy evals, failure matrices, and config variants produce comparable JSON/CSV/HTML artifacts.

The current environment is intentionally narrow: one arm, one tabletop reaching task, static PNG replay, and no hardware control path. The contact/friction fields are kept explicit so the reports do not imply physical safety metrics that the task does not yet measure.

## Repo layout

| Path | Role |
| --- | --- |
| `train.py` | Train a PPO policy from robot/task/reward YAML configs. |
| `eval.py` | Evaluate one policy with optional failure modes and a named gate profile. |
| `eval_matrix.py` | Run one policy across a YAML matrix preset and write ranked JSON/CSV/HTML outputs. |
| `sweep.py` | Train/evaluate named reward or task config variants and rank them. |
| `replay_gallery.py` | Build a static replay gallery from a matrix summary and replay index. |
| `run_index.py` | Index local train/eval/matrix/sweep artifact directories into JSON and CSV. |
| `kineforge/` | Environment, config loading, gates, evaluation, reports, sweeps, replay, gallery, registry helpers. |
| `configs/` | Robot, task, reward, failure, gate, matrix preset, and sweep preset YAML. |
| `examples/results/v0.6.0/` | Committed example matrix summary, CSV, HTML report, replay gallery, scorecards, and trajectory PNGs. |
| `RESULTS.md` | Reproduction commands and metric summary for the committed example artifacts. |

## Stack

- MuJoCo
- Gymnasium
- Stable-Baselines3 PPO
- NumPy
- PyYAML
- matplotlib
- pytest

## Credits

Built with MuJoCo, Gymnasium, Stable-Baselines3, NumPy, PyYAML, matplotlib, and pytest.

MIT licensed.

Contributor and agent workflow notes live in `PROJECT_DOCTRINE.md`, `ROADMAP.md`, and `AGENTS.md`.
