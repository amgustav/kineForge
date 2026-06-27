# kineForge

**Train, evaluate, and stress-test small MuJoCo robot policies.**

kineForge trains a PPO policy on a tabletop reaching task, runs it through configurable failures, gate profiles, eval matrices, and config sweeps, then writes local scorecards, reports, replay plots, and run indexes.

## What it does

- trains a PPO robot policy
- evaluates under failure modes and gate profiles
- runs eval matrices and config sweeps
- writes scorecards, HTML/CSV reports, replay plots, and run indexes

## Run

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
```

## Outputs

- scorecards
- matrix reports
- sweep reports
- replay plots and replay gallery
- run index

## Stack

- MuJoCo
- Gymnasium
- Stable-Baselines3 PPO
- NumPy
- PyYAML
- matplotlib
- pytest

## Files

- `train.py`
- `eval.py`
- `eval_matrix.py`
- `sweep.py`
- `replay_gallery.py`
- `run_index.py`
- `kineforge/`
- `configs/`
- `examples/results/v0.6.0/`
- `RESULTS.md`

## License

MIT.

Contributor and agent notes live in `PROJECT_DOCTRINE.md`, `ROADMAP.md`, and `AGENTS.md`.
