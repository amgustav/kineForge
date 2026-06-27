# kineForge Results

This file records the current public proof object for kineForge: a reproducible evaluation matrix for one trained tabletop reach policy.

## v0.6.0 Reproducible Results Capsule

The v0.6.0 capsule shows that kineForge can train a small MuJoCo robot arm policy, evaluate it under configured failure modes, apply a named gate profile, and write inspectable local artifacts.

Committed example artifacts live in [`examples/results/v0.6.0/`](examples/results/v0.6.0/):

- `matrix_summary.json` — machine-readable aggregate and per-scenario metrics.
- `summary.csv` — ranked scenario table.
- `report.html` — static local report.
- `replay_gallery.html` — static visual gallery linking ranked scenarios to trajectory plots and scorecards.
- `replay_index.json` — replay artifact index.
- `scenarios/*/scorecard.json` — per-scenario scorecards.
- `scenarios/*/trajectory.png` — selected replay plots.

The policy file is not committed. Reproduce it locally with the commands below.

## Reproduction commands

Environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Train the policy used for the result capsule:

```bash
python train.py --task tabletop_reach --robot arm_v0 --timesteps 25000 --seed 1
```

Run the eval matrix:

```bash
python eval_matrix.py --policy runs/latest/policy.zip --preset default --gate standard --episodes 3 --seed 1
```

Build or refresh the replay gallery:

```bash
python replay_gallery.py --summary examples/results/v0.6.0/matrix_summary.json --replay-index examples/results/v0.6.0/replay_index.json --output examples/results/v0.6.0/replay_gallery.html
```

The committed capsule was generated with:

- policy source: locally trained `runs/latest/policy.zip`
- robot: `arm_v0`
- task: `tabletop_reach`
- reward: `reach_v0`
- matrix preset: `default`
- gate profile: `standard`
- seed: `1`
- episodes per scenario: `3`
- training timesteps: `25000`

## Result summary

From `examples/results/v0.6.0/matrix_summary.json`:

| Metric | Value |
| --- | ---: |
| Scenarios | 7 |
| Gate profile | `standard` |
| Passing scenarios | 7 / 7 |
| Aggregate success rate | 1.000 |
| Aggregate mean final distance | 0.0451 m |
| Aggregate timeout rate | 0.000 |
| Aggregate collision rate | 0.000 |

Ranked scenarios:

| Rank | Scenario | Gate | Success rate | Mean final distance |
| ---: | --- | --- | ---: | ---: |
| 1 | `observation_noise` | PASS | 1.000 | 0.0399 |
| 2 | `baseline` | PASS | 1.000 | 0.0439 |
| 3 | `high_friction` | PASS | 1.000 | 0.0439 |
| 4 | `low_friction` | PASS | 1.000 | 0.0439 |
| 5 | `action_noise` | PASS | 1.000 | 0.0463 |
| 6 | `combined_hard` | PASS | 1.000 | 0.0480 |
| 7 | `target_shift` | PASS | 1.000 | 0.0495 |

## What this proves

- The train/eval loop runs end to end on the default tabletop reach task.
- A locally trained PPO policy can be evaluated deterministically across the default matrix preset.
- kineForge writes reproducible local artifacts: scorecards, matrix summary, CSV, HTML report, replay index, replay gallery, and trajectory plots.
- Gate profiles are applied and recorded in the scorecards and matrix summary.
- v0.8.0 scorecards explicitly mark collision/contact metrics as unmeasured placeholders and record whether active failure modes are physically modeled.
- v0.9.0 adds `run_index.py` for local JSON/CSV indexing of train, eval, matrix, and sweep artifacts.

## What this does not prove

- It does not prove real robot deployment readiness.
- It does not prove broad task generality; this is one robot and one reaching task.
- It does not prove collision/contact safety. Collision rate is retained as an explicit unmeasured placeholder for the current kinematic task.
- It does not prove physically modeled friction effects. `low_friction` and `high_friction` are documented placeholders in the current kinematic task.
- It does not claim benchmark superiority over other robot learning systems.
