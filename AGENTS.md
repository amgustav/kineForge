# kineForge

Open-source RL-first embodied AI testbed for forging robot policies. Train, stress, evaluate, compare, and replay robot policies before real-world deployment.

## Current Goal v0

One robot arm. One tabletop environment. One task: reach to target. One editable reward config. Domain randomization. Failure injection. RL training from scratch. Eval gate. JSON scorecard. Simple replay visualization. Clear README. One-command quickstart.

## Stack

- Python 3.11+
- MuJoCo via mujoco Python bindings for simulation
- Gymnasium for RL environment interface
- Stable-Baselines3 or CleanRL for RL training
- YAML configs for robots, tasks, rewards, failures
- JSON for eval reports and scorecards
- matplotlib or simple HTML for replay/visualization

## Rules

- Tests must pass before saying done.
- Never touch .env files.
- Keep v0 minimal: one robot, one task, one reward, one loop.
- All configs are YAML files, not hardcoded.
- Training and eval are separate CLI commands.
- Every eval produces a JSON scorecard.
- No web app, no frontend, no API server in v0.
- No VLA/LLM-first design. This is RL-first.
- No Kubernetes, no microservices, no Docker in v0.
- No over-engineering. Ship the loop first.

## Non-Goals

- Not a metaverse.
- Not a VLA playground.
- Not an Isaac Sim replacement.
- Not a full web app.
- Not a startup. It is a repo first.
- Not a CAD editor.
- Not a synthetic data engine.

## CLI Shape

Target commands:

python train.py --task tabletop_reach --robot arm_v0
python eval.py --policy runs/latest/policy.pt --failures moved_object,distractor,low_friction

Example report:

success_rate: 93%
collision_rate: 1%
unsafe_action_rate: 0%
gate: PASS
