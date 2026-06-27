# kineForge v1.0.0 Readiness

Last updated: 2026-06-27
Current released baseline: v0.9.0

## Verdict

kineForge is close to a stable local v1.0.0 surface, but should not be released as v1.0.0 until one final documentation and command-stability pass is complete.

Recommended v1.0.0 scope: stabilize what exists. Do not add a new robot, task family, web UI, database, cloud tracker, video system, or broad benchmark suite before v1.0.0.

## Current stable surface

Core commands:

```bash
python train.py --task tabletop_reach --robot arm_v0 --timesteps 1000 --seed 1
python eval.py --policy runs/latest/policy.zip --seed 1
python eval_matrix.py --policy runs/latest/policy.zip --preset default --seed 1
python sweep.py --preset default --timesteps 1000 --seed 1
python replay_gallery.py --summary examples/results/v0.6.0/matrix_summary.json --replay-index examples/results/v0.6.0/replay_index.json --output examples/results/v0.6.0/replay_gallery.html
python run_index.py --runs-dir runs --output runs/run_index.json --csv runs/run_index.csv
```

Released milestones:

- v0.5.0: configurable presets and gate profiles.
- v0.6.0: reproducible results capsule.
- v0.7.0: static replay gallery.
- v0.8.0: explicit physical metric metadata.
- v0.9.0: local run index.

## What is strong enough for v1.0.0

- One-command local training and evaluation loop exists.
- Evaluation writes scorecards, metadata, config snapshots, and replay plots.
- Matrix and sweep workflows write JSON, CSV, and HTML artifacts.
- Gate profiles are named, configurable, and recorded in outputs.
- Physical/contact placeholders are explicitly marked as unmeasured.
- Example result artifacts are committed under `examples/results/v0.6.0/`.
- Local run indexing exists without adding a database or cloud tracker.
- Tests cover core environment behavior, configs, gates, scorecards, matrices, sweeps, plots, gallery generation, and registry output.

## What remains weak

- The install path has not been tested from a fresh clone in this session.
- Collision/contact metrics remain unmeasured placeholders for the current kinematic task.
- Friction modes are documented but not physically modeled.
- Replay is PNG/static HTML only; no video output.
- Only one robot and one tabletop reaching task exist.
- Generated run directories are local artifacts and are not all committed.

## Required v1.0.0 release checklist

Before tagging v1.0.0:

1. Run the full test suite:

   ```bash
   .venv/bin/python -m pytest -q
   ```

2. Run one minimal smoke sequence:

   ```bash
   .venv/bin/python train.py --task tabletop_reach --robot arm_v0 --timesteps 1000 --seed 1
   .venv/bin/python eval.py --policy runs/latest/policy.zip --seed 1 --episodes 1 --gate smoke
   .venv/bin/python eval_matrix.py --policy runs/latest/policy.zip --preset default --gate smoke --episodes 1 --seed 1
   .venv/bin/python sweep.py --preset default --timesteps 1 --seed 1 --episodes 1 --gate smoke
   .venv/bin/python run_index.py --runs-dir runs --output runs/run_index.json --csv runs/run_index.csv
   ```

3. Confirm expected artifacts exist:

   - `runs/latest/policy.zip`
   - `runs/latest/scorecard.json`
   - latest `runs/eval-matrix-*/matrix_summary.json`
   - latest `runs/eval-matrix-*/report.html`
   - latest `runs/sweep-*/sweep_summary.json`
   - `runs/run_index.json`
   - `runs/run_index.csv`

4. Review README commands against the current CLI.
5. Confirm `git status --short` is clean.
6. If all checks pass, create annotated tag `v1.0.0` and a GitHub release.

## Non-goals before v1.0.0

- No new robot.
- No new task family.
- No dashboard, web server, database, cloud backend, or SaaS tracker.
- No claims of real-world deployment readiness.
- No claims of contact/collision safety until those metrics are actually measured.

## Recommendation

Proceed to v1.0.0 only after the checklist above passes from the current repo state. The release should be framed as a small, local, reproducible robot policy evaluation testbed, not as a broad robotics benchmark or deployment safety system.
