# kineForge Project Doctrine

Project-specific doctrine for `kineForge`.

Last updated: 2026-06-27
Current repository baseline: v0.7.0

This document is the project-specific operating doctrine for kineForge. It supplements the general operating doctrine and applies only to this repository.

It is written for humans first and autonomous coding agents second. It should be read together with:

- `AGENTS.md`
- `README.md`
- `ROADMAP.md`

## 1. Purpose

The purpose of this document is to make kineForge agent-friendly, reproducible, and strategically coherent over many future development sessions.

It defines:

- the mission of the project
- the engineering philosophy
- the architecture philosophy
- the autonomous development workflow
- the versioning and release policy
- the testing and verification standard
- the artifact standard
- the scope boundaries and non-goals
- how OMP or any future coding agent should behave in this repository

This document should prevent the project from drifting into vague platform-building, over-engineered architecture, or unrelated robotics fantasies.

## 2. Mission

kineForge is an RL-first, open-source robot policy testbed.

Its job is to help developers train, stress-test, evaluate, compare, replay, and report on robot policies in simulation before any real-world deployment work.

The project should emphasize:

- reproducibility
- evaluation quality
- inspectable artifacts
- local-first execution
- simple commands
- explicit configs
- deterministic testing where possible
- incremental engineering
- honest limitations

The project should not try to be impressive through size. It should be impressive through clarity, evidence, artifacts, and discipline.

## 3. Public Positioning

The public positioning of kineForge is:

> RL-first robot policy testbed: train, stress-test, evaluate, compare, and replay MuJoCo robot policies.

Preferred public language:

- robot policy evaluation
- simulation testbed
- failure matrices
- config sweeps
- scorecards
- replay artifacts
- reproducible experiment runs
- deployment-readiness preparation
- evaluation gates
- experiment reports

Avoid public language that is accusatory, conspiratorial, or unserious.

Do not frame kineForge as an exposé, manifesto, anti-industry project, or attack on robotics companies. The project is infrastructure, not commentary.

## 4. Current Baseline

As of v0.7.0, kineForge contains:

- a MuJoCo tabletop reach environment
- PPO training through Stable-Baselines3
- Gymnasium environment interface
- YAML configs for robot, task, reward, randomization, failures, gates, matrix presets, and sweep presets
- named gate profiles
- deterministic evaluation
- JSON scorecards with gate explanations
- timestamped run directories
- config snapshots
- configurable eval matrices
- replay indexes
- static HTML and CSV reports
- matrix comparison tooling
- config sweep tooling
- ranked matrix and sweep summaries
- tests covering environment behavior, configs, gates, scorecards, matrices, plots, and sweep logic
- reproducible results capsule with committed example matrix artifacts
- static replay gallery generation from matrix artifacts

The current project is intentionally small:

- one robot
- one task
- one reward config family
- one training loop
- one eval gate
- reproducible local artifacts

This smallness is a strength. Do not destroy it casually.

## 5. Architecture Philosophy

Prefer small, composable modules.

The architecture should remain readable to a new contributor and navigable by an autonomous coding agent.

Good architecture in kineForge means:

- clear file boundaries
- small functions
- explicit configs
- no hidden global behavior
- local run artifacts
- predictable output directories
- testable units
- backwards-compatible CLI behavior
- simple Python modules over framework-heavy architecture

Bad architecture in kineForge means:

- vague plugin systems before they are needed
- premature abstractions
- giant frameworks
- web apps before core evaluation improves
- databases before local artifacts are exhausted
- cloud infrastructure before local reproducibility is strong
- multi-robot complexity before one robot/task is deeply evaluated

Prefer extending existing systems over replacing them.

Rewrite only when:

- the current structure blocks the next milestone
- tests can preserve behavior
- the replacement is smaller, clearer, and more reproducible

## 6. RL-First Philosophy

kineForge is RL-first.

The core loop is:

```text
robot -> task -> reward -> train -> failures -> eval -> scorecard -> replay -> compare
```

Do not turn the project into:

- a VLA playground
- an LLM-agent demo
- a general autonomy manifesto
- a web app
- a synthetic data platform
- an Isaac Sim replacement
- a robot zoo
- a startup landing-page project

LLMs may help with development workflow, documentation, planning, and code generation. They are not the core runtime of v0.x.

## 7. Evaluation Doctrine

Evaluation is the heart of kineForge.

Every meaningful capability should eventually answer:

- What policy was evaluated?
- Under what config?
- Under what failure modes or scenario variations?
- With what seed?
- With how many episodes?
- What did it output?
- Did it pass the gate?
- Where are the scorecards?
- Where are the replay artifacts?
- What changed compared to another run?

Evaluation should produce machine-readable artifacts first:

- JSON scorecards
- JSON summaries
- CSV summaries
- static HTML reports
- plots or replay outputs
- config snapshots
- metadata

The project should prefer boring, inspectable, local outputs over impressive-looking dashboards.

A dashboard is not a priority unless it reveals something that the JSON/CSV/HTML report cannot.

## 8. Experiment Doctrine

A kineForge experiment should be reproducible.

Every experiment should capture:

- policy path
- robot config
- task config
- reward config
- failure config
- sweep or matrix config
- seed
- timesteps
- episodes
- output directory
- generated scorecards
- generated reports

Experiments should be easy to rerun locally.

A good experiment command looks like:

```bash
python train.py --task tabletop_reach --robot arm_v0 --timesteps 25000 --seed 1
python eval.py --policy runs/latest/policy.zip --seed 1
python eval_matrix.py --policy runs/latest/policy.zip --seed 1
python sweep.py --config configs/sweeps/default.yaml --timesteps 1000 --seed 1
```

## 9. Artifact Doctrine

kineForge exists through artifacts.

Every major feature should produce something inspectable:

- scorecard
- summary file
- report
- CSV
- plot
- replay
- config snapshot
- metadata file
- comparison output

If a feature does not improve training, evaluation, reproducibility, comparison, inspection, or documentation, it is probably not a priority.

The output is the product.

## 10. Reproducibility Doctrine

Prefer explicit seeds, explicit configs, and timestamped output directories.

A run should not depend on hidden state unless documented.

Generated artifacts should be easy to locate, inspect, and compare.

The `runs/` directory is the local experimental record.

Every new experiment system should answer:

- Where does the output go?
- What file is the machine-readable summary?
- What file is the human-readable report?
- What config produced it?
- What policy produced it?
- How do I compare it with another run?

## 11. Autonomous Development Doctrine

The objective is milestone autonomy, not infinite autonomy.

An autonomous session should have a bounded target:

- v0.5.0
- v0.6.0
- v0.7.0
- v1.0.0

The agent may create intermediate patch versions as needed:

- v0.4.1
- v0.4.2
- v0.4.3

But it should not invent an unrelated direction.

The standard autonomous loop is:

```text
read AGENTS.md
read PROJECT_DOCTRINE.md
read ROADMAP.md
read README.md
inspect architecture
plan next improvement
implement
run tests
fix failures
verify artifacts
update docs
commit
push
continue
```

Repeat until the requested target version is complete or a real stop condition is reached.

## 12. Agent Behavior Rules

When using OMP or a similar agent:

- Read `AGENTS.md`, `PROJECT_DOCTRINE.md`, `ROADMAP.md`, and `README.md` before editing code.
- Use the existing architecture unless a change is necessary.
- Do not ask for approval for routine implementation choices.
- Do ask or stop for architectural ambiguity that affects project direction.
- Run tests before every commit.
- Update docs when behavior changes.
- Commit and push after every verified meaningful improvement.
- Keep the repository clean before stopping.
- Do not create tags or GitHub Releases except at meaningful minor milestones unless explicitly instructed.

The agent should be autonomous within the milestone, not autonomous in choosing the project identity.

## 13. Versioning Policy

Use semantic-style pre-1.0 versioning:

### Patch versions

Examples:

- v0.4.1
- v0.4.2
- v0.5.1

Patch versions are meaningful improvements within the current capability band.

They may include:

- docs improvements
- CLI polish
- report improvements
- bug fixes
- test improvements
- small artifact improvements
- minor backwards-compatible config additions

Patch versions require:

- tests passing
- docs updated if needed
- commit
- push

Patch versions do not always require GitHub Releases unless the user explicitly wants one.

### Minor versions

Examples:

- v0.5.0
- v0.6.0
- v0.7.0

Minor versions are coherent new capability milestones.

They require:

- tests passing
- smoke verification
- expected artifacts generated
- README updated
- commit
- push
- annotated git tag
- GitHub Release

### Major versions

v1.0.0 requires explicit human review.

Do not let an autonomous agent publish v1.0.0 without human approval.

## 14. Git Workflow

Before work:

```bash
git status
```

The repository should normally be clean before an autonomous session begins.

After each verified improvement:

```bash
git add -u
git add <new intended files>
git commit -m "clear message"
git push
```

Commit messages should be short and descriptive:

- `docs: add project doctrine and roadmap`
- `feat: add configurable matrix presets`
- `feat: add stricter eval gates`
- `fix: correct sweep ranking order`
- `test: cover matrix preset loading`

Do not commit generated run artifacts unless they are intentionally added as examples.

Do not commit `.env`, credentials, local secrets, caches, or accidental junk.

## 15. Release Policy

Create a GitHub Release only for meaningful milestones.

Release notes should include:

- what changed
- what was added
- what was preserved
- what was verified
- current limitations if relevant

Do not overhype release notes.

A release is an engineering checkpoint, not marketing fluff.

## 16. Testing Doctrine

Every meaningful change must run tests.

Default test command:

```bash
.venv/bin/python -m pytest -q
```

If system Python fails because dependencies are not installed globally, that is not a blocker. The repository's `.venv` is the valid project environment.

Tests should cover:

- config loading
- environment behavior
- scorecard generation
- eval matrix behavior
- report generation
- comparison logic
- sweep ranking logic
- regression-prone utilities

Prefer deterministic tests over fragile visual tests.

## 17. Documentation Doctrine

Documentation should stay close to reality.

The README should explain:

- what works now
- how to run it
- what outputs are generated
- how to inspect artifacts
- current limitations
- roadmap

`PROJECT_DOCTRINE.md` explains how the project should evolve.

`ROADMAP.md` explains where the project is going.

`AGENTS.md` gives practical rules for coding agents.

Do not put private strategy, speculation, or external industry commentary in public repo docs.

## 18. Non-Goals

kineForge is not:

- a full robotics platform
- a web app
- a dashboard-first product
- a cloud service
- a Kubernetes project
- a microservices architecture
- a startup pitch deck
- a VLA playground
- an LLM-first robotics agent
- a CAD editor
- an Isaac Sim replacement
- a robot zoo
- a synthetic data engine
- a hardware deployment framework
- a real robot control stack
- a weapons/fire-control project

These non-goals can be revised later, but only deliberately.

## 19. Stop Conditions

An autonomous agent should stop only if:

- external credentials are required
- a human architectural decision is genuinely required
- tests cannot be made to pass after serious debugging
- a destructive operation outside the repository would be required
- the requested milestone is complete
- the task would violate project non-goals
- secrets or credentials are encountered

Routine implementation uncertainty is not a stop condition.

## 20. Scope Control

When in doubt, choose the smaller, more inspectable version.

Prefer:

- one good config system over three half-built systems
- one clear report over a dashboard
- one reliable task over five toy tasks
- one reproducible experiment over a vague benchmark suite
- one useful CLI over a web server

Delete before adding.

## 21. Roadmap Interpretation Rules

The roadmap defines outcomes, not exact implementation.

Agents may choose implementation details if they preserve:

- architecture simplicity
- backwards compatibility
- local reproducibility
- artifact quality
- test coverage
- scope boundaries

If a roadmap item is too large, split it into patch versions.

Do not skip ahead to later roadmap items unless they directly unblock the current milestone.

## 22. Public Repository Standard

The public repository should be legible in 10 seconds.

A visitor should quickly understand:

- what kineForge is
- how to install it
- how to train
- how to evaluate
- what artifacts it writes
- what current limitations exist
- what is next

Prefer professional, restrained language.

Do not overstate capabilities.

Do not hide limitations.

## 23. Future Expansion Principles

Future expansion should proceed from the current core:

1. better configs
2. better evals
3. better gates
4. better metrics
5. better replay/inspection
6. broader tasks
7. broader robots
8. stronger experiment management
9. stable v1.0 release

Do not jump directly to a broad platform.

## 24. Default OMP Prompt Pattern

Use this pattern for long autonomous sessions:

```text
/goal Continue autonomously until v0.X.0 is complete.

Read:
- AGENTS.md
- PROJECT_DOCTRINE.md
- ROADMAP.md
- README.md

Follow the project doctrine and roadmap.

After every verified improvement:
- run tests
- fix failures
- update documentation if needed
- create one atomic commit
- push to origin
- continue automatically

For patch versions:
- commit
- push

For minor versions:
- commit
- push
- create annotated tag
- create GitHub Release
- stop

Stop only if a defined stop condition is reached.

Before stopping, report completed versions, features, verification results, latest commit hash, and git status.
```

## 25. Final Rule

kineForge should become a serious, reproducible, inspectable robot policy testbed through disciplined increments.

Do not make it bigger to make it look impressive.

Make it sharper.
