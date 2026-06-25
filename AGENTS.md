# kineForge

Open-source RL-first embodied AI testbed for forging robot policies. Train, stress, evaluate, compare, and replay robot policies before real-world deployment.

## Current Goal v0

One simple robot arm. One tabletop environment. One task: reach to target. One editable reward config. Domain randomization. Failure injection. PPO training from scratch. Eval gate. JSON scorecard. Simple replay visualization. Clear README. One-command quickstart.

## Locked v0 Technical Choices

- Simulator: MuJoCo via the `mujoco` Python bindings.
- RL environment interface: Gymnasium.
- RL library: Stable-Baselines3.
- RL algorithm: PPO.
- Robot: simple Reacher-style 2-DoF or 3-DoF arm.
- Task: tabletop reach-to-target.
- Observation: joint positions, joint velocities, end-effector position, target position, and relative end-effector-to-target vector.
- Action: continuous joint control.
- Reward: negative distance to target, success bonus, control/action penalty, timeout penalty.
- Domain randomization: target position, initial joint positions, observation noise, actuator/control scale.
- Failure modes: moved_target, noisy_observation, weak_actuator.
- Replay: save a trajectory PNG first. GIF/video only if easy.
- Output: `runs/latest/policy.zip`, `runs/latest/scorecard.json`, and a timestamped run folder.

## Stack

- Python 3.11+
- MuJoCo via mujoco Python bindings for simulation
- Gymnasium for RL environment interface
- Stable-Baselines3 PPO for RL training
- YAML configs for robots, tasks, rewards, failures
- JSON for eval reports and scorecards
- matplotlib for replay/visualization
- pytest for smoke tests if useful

## Rules

- Tests or smoke checks must pass before saying done.
- Never touch .env files.
- Keep v0 minimal: one robot, one task, one reward, one loop.
- All configs are YAML files, not hardcoded.
- Training and eval are separate CLI commands.
- Every eval produces a JSON scorecard.
- No web app, no frontend, no API server in v0.
- No VLA/LLM-first design. This is RL-first.
- No Kubernetes, no microservices, no Docker in v0.
- No over-engineering. Ship the loop first.
- Prefer simple, working code over abstract architecture.
- Do not add cloud, auth, database, UI framework, or services.

## Non-Goals

- Not a metaverse.
- Not a VLA playground.
- Not an Isaac Sim replacement.
- Not a full web app.
- Not a startup. It is a repo first.
- Not a CAD editor.
- Not a synthetic data engine.
- Not a humanoid robotics project in v0.
- Not a benchmark suite in v0.

## CLI Shape

Target commands:

python train.py --task tabletop_reach --robot arm_v0 --timesteps 1000

python eval.py --policy runs/latest/policy.zip --failures moved_target,noisy_observation,weak_actuator

Example report:

success_rate: 93%
mean_final_distance: 0.04
timeout_rate: 0.05
collision_rate: 0.0
gate: PASS

## Acceptance Criteria

- `python train.py --task tabletop_reach --robot arm_v0 --timesteps 1000` runs without crashing.
- `python eval.py --policy runs/latest/policy.zip --failures moved_target,noisy_observation,weak_actuator` runs without crashing.
- Eval writes `runs/latest/scorecard.json`.
- Replay writes a trajectory PNG.
- README explains install, train, eval, outputs, limitations, and roadmap.
- `git status` is clean or remaining changes are clearly explained.
