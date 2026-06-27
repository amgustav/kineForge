# kineForge

kineForge is an open-source RL-first embodied AI testbed for robot policy training, stress testing, evaluation, comparison, and replay.

## Required Reading

Every future agent session must read these files before editing:

- `AGENTS.md`
- `PROJECT_DOCTRINE.md`
- `ROADMAP.md`
- `README.md`

## v0 Goal

Build the smallest useful robot-learning loop:

- one simple MuJoCo robot arm
- one tabletop reach-to-target task
- one editable reward config
- domain randomization
- failure injection
- PPO training from scratch
- deterministic evaluation
- JSON scorecard
- simple trajectory replay
- clear README
- one-command quickstart

## Locked v0 Choices

- Simulator: MuJoCo via Python bindings
- RL interface: Gymnasium
- RL library: Stable-Baselines3
- Algorithm: PPO
- Robot: simple Reacher-style 2-DoF or 3-DoF arm
- Task: tabletop reach-to-target
- Configs: YAML
- Reports: JSON
- Replay: matplotlib trajectory PNG

## Implementation Guidelines

- Keep v0 minimal and working.
- Prefer simple, inspectable code over abstract architecture.
- Keep configs in YAML instead of hardcoding task/reward/failure values.
- Keep training and evaluation as separate CLI commands.
- Every evaluation run should produce a JSON scorecard.
- Add smoke tests or simple verification commands where useful.
- Do not add a web app, API server, database, cloud backend, Docker, Kubernetes, auth system, or frontend in v0.
- Do not build VLA/LLM-first robotics behavior in v0.

## Target CLI

python train.py --task tabletop_reach --robot arm_v0 --timesteps 1000

python eval.py --policy runs/latest/policy.zip --failures moved_target,noisy_observation,weak_actuator

## Acceptance Criteria

- Training runs without crashing.
- Evaluation runs without crashing.
- Eval writes runs/latest/scorecard.json.
- Replay writes a trajectory PNG.
- README explains install, train, eval, outputs, limitations, and roadmap.
