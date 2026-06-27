# kineForge

**Train, stress-test, and evaluate robot policies in simulation.**

kineForge is an open-source RL-first embodied AI testbed. It trains a small MuJoCo robot arm with PPO, evaluates it under configurable failure modes, and writes reproducible local run artifacts.

The repo is intentionally small: one robot, one task, one reward config, one training loop, one eval gate, JSON reports, and matplotlib PNG replay/diagnostic plots.

```text
robot → task → reward → train → failures → eval → scorecard → replay
```

---

## Project documentation

- [`PROJECT_DOCTRINE.md`](PROJECT_DOCTRINE.md) defines the project mission, engineering doctrine, scope boundaries, and autonomous workflow.
- [`ROADMAP.md`](ROADMAP.md) defines the milestone direction from the current baseline toward v1.0.0.
- [`AGENTS.md`](AGENTS.md) defines practical rules future coding agents must follow in this repository.

---

## Quickstart

```bash
git clone https://github.com/amgustav/kineForge.git
cd kineForge

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
python -m pytest tests/test_env_smoke.py -q
```

Train a smoke-test policy:

```bash
python train.py --task tabletop_reach --robot arm_v0 --timesteps 1000 --seed 1
```

Train the recommended local learning run:

```bash
python train.py --task tabletop_reach --robot arm_v0 --timesteps 25000 --seed 1
```

Evaluate the normal no-failure gate:

```bash
python eval.py --policy runs/latest/policy.zip --seed 1
```

Stress-test it with failure modes:

```bash
python eval.py --policy runs/latest/policy.zip --failures moved_target,noisy_observation,weak_actuator --seed 1
```

Run the default eval matrix. This evaluates the same policy in `baseline`, `target_shift`, `low_friction`, `high_friction`, `observation_noise`, `action_noise`, and `combined_hard` scenarios:

```bash
python eval_matrix.py --policy runs/latest/policy.zip --seed 1
```

Run a custom named eval matrix:

```bash
python eval_matrix.py --policy runs/latest/policy.zip --scenario baseline= --scenario target_shift=moved_target --scenario observation_noise=noisy_observation --scenario action_noise=action_noise --scenario combined_hard=moved_target,noisy_observation,action_noise,weak_actuator --seed 1
```

Compare two eval matrix summaries:

```bash
python compare_eval.py --before runs/eval-matrix-YYYYMMDD-HHMMSS/matrix_summary.json --after runs/eval-matrix-YYYYMMDD-HHMMSS/matrix_summary.json --output runs/matrix_comparison.json
```

Current status: kineForge has the MuJoCo tabletop reach environment, PPO training, YAML configs, deterministic eval, JSON scorecards, trajectory PNGs, timestamped runs, explicit seeds, metadata, config snapshots, eval matrices, config sweeps, replay indexes, and matrix/sweep summary reports.

Outputs:

```text
runs/
  train-YYYYMMDD-HHMMSS/
    policy.zip
    train_metadata.json
    config_snapshot.yaml
  eval-YYYYMMDD-HHMMSS/
    policy.zip
    scorecard.json
    eval_metadata.json
    config_snapshot.yaml
    trajectory.png
    distance_over_time.png
    episode_rewards.png
  eval-matrix-YYYYMMDD-HHMMSS/
    policy.zip
    train_metadata.json        # present when the evaluated policy came from a kineForge train run
    matrix_summary.json
    replay_index.json
    report.html
    summary.csv
    scenarios/
      baseline/
        scorecard.json
        eval_metadata.json
        config_snapshot.yaml
        trajectory.png
        distance_over_time.png
        episode_rewards.png
      combined_hard/
        scorecard.json
        eval_metadata.json
        config_snapshot.yaml
        trajectory.png
        distance_over_time.png
        episode_rewards.png
  sweep-YYYYMMDD-HHMMSS/
    sweep_summary.json
    sweep_report.html
    summary.csv
    variants/
      baseline/
        policy.zip
        scorecard.json
        eval_metadata.json
        config_snapshot.yaml
  latest/
    policy.zip
    scorecard.json
    train_metadata.json        # present when the evaluated policy came from a kineForge train run
    eval_metadata.json
    config_snapshot.yaml
    trajectory.png
    distance_over_time.png
    episode_rewards.png
```

---

## What it does

|                     |                                                              |
| ------------------- | ------------------------------------------------------------ |
| **Robot**           | 2-DoF Reacher-style MuJoCo arm                               |
| **Task**            | tabletop reach-to-target                                     |
| **Training**        | PPO from scratch via Stable-Baselines3                       |
| **Environment API** | Gymnasium                                                    |
| **Configs**         | YAML robot, task, reward, randomization, and failure configs |
| **Failure modes**   | moved target, noisy observation, action noise, weak actuator; low/high friction are documented matrix placeholders |
| **Eval output**     | JSON scorecards, matrix summaries, replay indexes, and eval metadata |
| **Replay output**   | matplotlib trajectory, distance, and reward PNGs             |
| **Tests**           | smoke tests for env behavior, configs, scorecards, matrices, and plots |

A short `1000` timestep run is a smoke test. It proves the pipeline works; the `25000` timestep command is the recommended local run expected to pass normal no-failure eval.

---

## Scorecard

Evaluation writes a machine-readable scorecard with:

* `summary`: success rate, mean final distance, timeout rate, mean episode reward, and collision rate.
* `gate.status`: `PASS` only when all gate criteria pass.
* `gate.thresholds`: min success rate, max mean final distance, and max timeout rate.
* `per_episode`: seed, success, timeout, final distance, episode reward, step count, target/final positions, and active failures for each episode.
* `failure_modes`: sorted failure modes active during evaluation.
* `artifacts`: paths to the policy snapshot, scorecard, metadata, config snapshot, and PNG plots.
* `collision_rate_explanation`: v0 does not implement collision detection, so `collision_rate` is fixed at `0.0` and is not a safety metric.

The eval gate separates two different things:

* did the code run?
* did the policy actually satisfy the task threshold?

A failed gate after short training means the policy did not pass yet, not that the repo is broken.

---

## Eval matrix

`eval_matrix.py` runs one policy snapshot across multiple named scenarios. Each scenario uses the same robot, task, reward config, seed, and episode count, with only the scenario failure set changing.

Scenario syntax is `name=failure_a,failure_b`. Use `name=` for a no-failure scenario. If no scenarios are provided, the default matrix runs:

* `baseline=` — no injected failures.
* `target_shift=moved_target` — moved target offset.
* `low_friction=low_friction` — documented limitation; not physically modeled in v0.
* `high_friction=high_friction` — documented limitation; not physically modeled in v0.
* `observation_noise=noisy_observation` — Gaussian observation noise.
* `action_noise=action_noise` — Gaussian action perturbation.
* `combined_hard=moved_target,noisy_observation,action_noise,weak_actuator` — combined compatible stressors. It intentionally excludes friction placeholders until friction is physically modeled.

Each matrix run writes:

* one scenario directory per scenario under `scenarios/<name>/`;
* one `scorecard.json` per scenario;
* one aggregate `matrix_summary.json`;
* one `replay_index.json` mapping scenario names to replay PNG artifacts that were written;
* one static `report.html` for local inspection;
* one `summary.csv` for spreadsheet, CLI, or downstream analysis.

Open the static matrix report locally after a run:

```bash
python eval_matrix.py --policy runs/latest/policy.zip --seed 1
open runs/eval-matrix-YYYYMMDD-HHMMSS/report.html
```

The matrix output directory shape is:

```text
runs/eval-matrix-YYYYMMDD-HHMMSS/
  matrix_summary.json
  replay_index.json
  report.html
  summary.csv
  scenarios/
    <scenario>/
      scorecard.json
```

`report.html` is a dependency-free local summary for quick inspection. `summary.csv` has one row per scenario and can be opened in a spreadsheet, inspected with command-line tools, or imported into downstream analysis.

`compare_eval.py` compares two `matrix_summary.json` files and reports aggregate and per-scenario metric deltas.

---

## Config sweeps

`sweep.py` trains and evaluates multiple named task/reward config variants from one YAML file, then ranks them by gate status, success rate, and mean final distance.

```bash
python sweep.py --config configs/sweeps/default.yaml --timesteps 1000 --seed 1
```

Each sweep writes one timestamped output directory:

```text
runs/sweep-YYYYMMDD-HHMMSS/
  sweep_summary.json
  sweep_report.html
  summary.csv
  variants/
    <variant_name>/
      policy.zip
      scorecard.json
      eval_metadata.json
      config_snapshot.yaml
```

Use `sweep_summary.json` for machine-readable ranking, `summary.csv` for spreadsheet/CLI inspection, and `sweep_report.html` for a dependency-free local report.

---


## How it works

`TabletopReachEnv` wraps a simple MuJoCo arm as a Gymnasium environment.

Observations include joint state, end-effector position, target position, and the relative vector to the target. Actions are continuous joint controls.

Training is handled by `train.py` using Stable-Baselines3 PPO.

Evaluation is handled by `eval.py`, which snapshots the policy, can inject configured failures, then writes a scorecard, metadata, config snapshot, trajectory plot, distance plot, and episode reward plot. Eval matrices are handled by `eval_matrix.py`, config sweeps by `sweep.py`, and summary comparison by `compare_eval.py`.

Reward terms are loaded from YAML and include distance-to-target, progress shaping, success bonus, control penalty, and timeout penalty.

---

## Repo layout

```text
configs/
  failures/basic_failures.yaml
  rewards/reach_v0.yaml
  robots/arm_v0.yaml
  sweeps/default.yaml
  tasks/tabletop_reach.yaml
kineforge/
  envs/tabletop_reach_env.py
  config.py
  evals.py
  eval_artifacts.py
  randomization.py
  matrix.py
  sweeps.py
  replay.py
  reports.py
  rewards.py

tests/
  test_env_smoke.py

train.py
eval.py
eval_matrix.py
compare_eval.py
sweep.py
```

---

## Current limitations

* one simple robot arm
* one tabletop reaching task
* basic failure injection
* no collision detection
* no real robot deployment
* PNG plots only
* no web, cloud, database, or backend

---

## Roadmap

Next experiment-quality steps:

* configurable sweep presets
* configurable matrix presets
* stricter eval gates
* real collision/contact metrics
* optional video replay

---

## Credits

Built with MuJoCo, Gymnasium, Stable-Baselines3, NumPy, PyYAML, matplotlib, and pytest.

## License

MIT
