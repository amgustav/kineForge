# kineForge

kineForge is an open-source, RL-first embodied AI testbed for robot policy training, stress testing, evaluation, comparison, and replay.

## v0 scope

v0 implements the smallest useful robot-learning loop:

- MuJoCo-backed 2-DoF Reacher-style tabletop arm.
- Gymnasium environment for tabletop reach-to-target.
- YAML configs for robot, task, reward, domain randomization, and eval failures.
- Stable-Baselines3 PPO training from scratch.
- Deterministic evaluation with optional failure injection.
- JSON scorecard and matplotlib trajectory PNG replay.
- Top-level CLI scripts for train and eval.

v0 deliberately does not include a web app, API server, Docker, Kubernetes, CAD editor, VLA/LLM system, cloud backend, database, auth system, or frontend.

## Install

Requires Python 3.11+ and a local MuJoCo-compatible runtime.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Train

```bash
python train.py --task tabletop_reach --robot arm_v0 --timesteps 1000
```

This writes:

- `runs/latest/policy.zip`
- `runs/latest/training_metadata.json`
- `runs/latest/config_snapshot.yaml`
- timestamped copies under `runs/train-*`

## Evaluate

```bash
python eval.py --policy runs/latest/policy.zip --failures moved_target,noisy_observation,weak_actuator
```

This writes:

- `runs/latest/scorecard.json`
- `runs/latest/trajectory.png`
- timestamped copies under `runs/eval-*`

The eval gate reports `PASS` or `FAIL`. A 1000-step training run is a smoke run and may produce a truthful `FAIL` gate.

## Scorecard shape

Example values below are illustrative; keys match the emitted v0 scorecard.

```json
{
  "collision_note": "Collision detection is not implemented in v0; collision_rate is fixed at 0.0.",
  "collision_rate": 0.0,
  "episodes": 10,
  "failures": ["moved_target", "noisy_observation", "weak_actuator"],
  "gate": {
    "criteria": {
      "mean_final_distance <= 0.05": false,
      "success_rate >= 0.80": false,
      "timeout_rate <= 0.30": false
    },
    "failed_criteria": [
      "success_rate >= 0.80",
      "mean_final_distance <= 0.05",
      "timeout_rate <= 0.30"
    ],
    "status": "FAIL"
  },
  "mean_episode_reward": -42.0,
  "mean_final_distance": 0.42,
  "policy_path": "runs/latest/policy.zip",
  "reward": "reach_v0",
  "robot": "arm_v0",
  "success_rate": 0.0,
  "task": "tabletop_reach",
  "timeout_rate": 1.0
}
```

## Limitations

- Simple 2-DoF arm only.
- No collision, obstacle, or distractor metrics; scorecards fix `collision_rate` at `0.0` and include an explanatory `collision_note`.
- No video or GIF replay; v0 writes a single trajectory PNG.
- Short 1000-step PPO runs are for smoke testing and may not learn a passing policy.
- MuJoCo must work on the local machine.

## Roadmap

- Richer robots and task suites.
- Collision, obstacle, and distractor metrics.
- More evaluation suites and failure modes.
- Richer replay outputs.
