# kineForge

**Train, stress-test, and evaluate robot policies in simulation.**

kineForge is an open-source RL-first embodied AI testbed. It trains a small MuJoCo robot arm with PPO, evaluates it under configurable failure modes, and writes reproducible local run artifacts.

The repo is intentionally small: one robot, one task, one reward config, one training loop, one eval gate, JSON reports, and matplotlib PNG replay/diagnostic plots.

```text
robot → task → reward → train → failures → eval → scorecard → replay
```

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

Current status: v0.1.0 established the MuJoCo tabletop reach environment, PPO training, YAML configs, deterministic eval, JSON scorecard, and trajectory PNG. v0.2.0 adds timestamped runs, explicit seeds, metadata, config snapshots, richer scorecards, and extra PNG plots.

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
| **Failure modes**   | moved target, noisy observation, weak actuator               |
| **Eval output**     | JSON scorecard and eval metadata                             |
| **Replay output**   | matplotlib trajectory, distance, and reward PNGs             |
| **Tests**           | smoke tests for env behavior, configs, scorecards, and plots |

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
* `collision_rate_explanation`: v0.2.0 does not implement collision detection, so `collision_rate` is fixed at `0.0` and is not a safety metric.

The eval gate separates two different things:

* did the code run?
* did the policy actually satisfy the task threshold?

A failed gate after short training means the policy did not pass yet, not that the repo is broken.

---

## How it works

`TabletopReachEnv` wraps a simple MuJoCo arm as a Gymnasium environment.

Observations include joint state, end-effector position, target position, and the relative vector to the target. Actions are continuous joint controls.

Training is handled by `train.py` using Stable-Baselines3 PPO.

Evaluation is handled by `eval.py`, which snapshots the policy, can inject configured failures, then writes a scorecard, metadata, config snapshot, trajectory plot, distance plot, and episode reward plot.

Reward terms are loaded from YAML and include distance-to-target, progress shaping, success bonus, control penalty, and timeout penalty.

---

## Repo layout

```text
configs/
  failures/basic_failures.yaml
  rewards/reach_v0.yaml
  robots/arm_v0.yaml
  tasks/tabletop_reach.yaml

kineforge/
  envs/tabletop_reach_env.py
  config.py
  evals.py
  randomization.py
  replay.py
  reports.py
  rewards.py

tests/
  test_env_smoke.py

train.py
eval.py
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

* config sweep runner
* run comparison helpers
* stricter eval gates
* real collision/contact metrics
* optional video replay

---

## Credits

Built with MuJoCo, Gymnasium, Stable-Baselines3, NumPy, PyYAML, matplotlib, and pytest.

## License

MIT
