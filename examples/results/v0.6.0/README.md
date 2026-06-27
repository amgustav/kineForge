# v0.6.0 result capsule

This directory contains a curated eval matrix output for the default tabletop reach policy evaluation.

Generated command:

```bash
python eval_matrix.py --policy runs/latest/policy.zip --preset default --gate standard --episodes 3 --seed 1
```

The policy file is intentionally not committed. Recreate it with:

```bash
python train.py --task tabletop_reach --robot arm_v0 --timesteps 25000 --seed 1
```

Contents:

- `matrix_summary.json` — aggregate and per-scenario metrics.
- `summary.csv` — ranked scenario table.
- `report.html` — static report generated from the committed summary and replay index.
- `replay_gallery.html` — static visual gallery generated from the matrix summary and replay index.
- `replay_index.json` — paths to committed trajectory plots.
- `scenarios/*/scorecard.json` — per-scenario scorecards.
- `scenarios/*/trajectory.png` — selected replay trajectories.


Refresh the gallery with:

```bash
python replay_gallery.py --summary examples/results/v0.6.0/matrix_summary.json --replay-index examples/results/v0.6.0/replay_index.json --output examples/results/v0.6.0/replay_gallery.html
```

Limitations:

- one robot and one tabletop reaching task;
- no real robot deployment claim;
- collision/contact metrics are placeholders;
- friction modes are documented but not physically modeled in this task.
