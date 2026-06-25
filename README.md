# kineForge

**Train, stress-test, and evaluate robot policies in simulation.**

kineForge is an open-source RL-first embodied AI testbed. The current version trains a small MuJoCo robot arm with PPO, evaluates it under configurable failure modes, and writes a JSON scorecard plus trajectory replay.

The repo is intentionally small: one robot, one task, one reward config, one training loop, one eval gate, one replay output.

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

Train a policy:

```bash
python train.py --task tabletop_reach --robot arm_v0 --timesteps 1000
```

Evaluate it with failure modes:

```bash
python eval.py --policy runs/latest/policy.zip --failures moved_target,noisy_observation,weak_actuator
```

Outputs:

```text
runs/latest/policy.zip
runs/latest/scorecard.json
runs/latest/trajectory.png
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
| **Eval output**     | JSON scorecard                                               |
| **Replay output**   | matplotlib trajectory PNG                                    |
| **Tests**           | basic smoke tests for env reset/step and config loading      |

A short `1000` timestep run is a smoke test. It proves the pipeline works; it is not expected to produce a strong policy.

---

## Scorecard

Evaluation writes a machine-readable scorecard:

```json
{
  "success_rate": 0.0,
  "mean_final_distance": 0.068,
  "timeout_rate": 1.0,
  "collision_rate": 0.0,
  "mean_episode_reward": -8.42,
  "gate": "FAIL"
}
```

The eval gate separates two different things:

* did the code run?
* did the policy actually satisfy the task threshold?

A failed gate after short training means the policy did not pass yet, not that the repo is broken.

---

## How it works

`TabletopReachEnv` wraps a simple MuJoCo arm as a Gymnasium environment.

Observations include joint state, end-effector position, target position, and the relative vector to the target. Actions are continuous joint controls.

Training is handled by `train.py` using Stable-Baselines3 PPO.

Evaluation is handled by `eval.py`, which can inject configured failures before writing a scorecard and replay image.

Reward terms are loaded from YAML and include distance-to-target, success bonus, control penalty, and timeout penalty.

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

kineForge is early.

Current limits:

* one simple robot arm
* one tabletop reaching task
* basic reward shaping
* basic failure injection
* basic trajectory replay
* no real robot deployment
* no benchmark result yet
* no web UI or cloud backend

---

## Roadmap

Next useful steps:

* make the reach policy learn reliably with better defaults
* add longer training recipes
* improve reward configs
* add experiment sweeps across reward and failure settings
* improve replay plots
* add more robot variants
* add more tasks
* add stricter eval gates

---

## Credits

Built with MuJoCo, Gymnasium, Stable-Baselines3, NumPy, PyYAML, matplotlib, and pytest.

## License

MIT
