# kineForge Roadmap

Last updated: 2026-06-27
Current baseline: v0.7.0

This roadmap defines the intended direction of kineForge from v0.7.0 to v1.0.0.

It is an outcome roadmap, not a step-by-step implementation script.

Agents should use this file together with:

- `AGENTS.md`
- `PROJECT_DOCTRINE.md`
- `README.md`

## Current State: v0.7.0

kineForge currently provides:

- MuJoCo tabletop reach environment
- PPO training via Stable-Baselines3
- Gymnasium environment interface
- YAML configs for robot, task, reward, failures, gates, randomization, matrix presets, and sweep presets
- named gate profiles
- deterministic evaluation
- JSON scorecards with gate explanations
- timestamped run directories
- configurable eval matrix presets
- configurable config sweep presets
- ranked matrix and sweep summaries
- static HTML and CSV matrix/sweep reports
- replay indexes
- matrix summary comparison
- reproducible results capsule with committed example matrix artifacts
- static replay gallery generation from matrix artifacts
- test coverage for core behavior

Current public positioning:

> RL-first robot policy testbed: train, stress-test, evaluate, and replay MuJoCo robot policies.

## Roadmap Principles

The roadmap follows this sequence:

1. Improve configurability.
2. Improve evaluation quality.
3. Improve metrics.
4. Improve replay and inspection.
5. Expand task/robot scope carefully.
6. Stabilize toward v1.0.0.

Do not skip directly to broad robotics platform work.

Do not add a web app, database, cloud architecture, or large framework unless the roadmap explicitly changes.

## v0.4.x - Documentation and Autonomous Workflow Foundation

### Objective

Make the repository easier for humans and autonomous agents to continue safely.

### Candidate improvements

- Add `PROJECT_DOCTRINE.md`.
- Add `ROADMAP.md`.
- Update `AGENTS.md` to require reading project doctrine and roadmap.
- Update README with project documentation links.
- Improve contributor/developer guidance.
- Clarify release/version workflow.

### Exit criteria

- `PROJECT_DOCTRINE.md` exists.
- `ROADMAP.md` exists.
- `AGENTS.md` points to both files.
- README explains the documentation hierarchy.
- Tests still pass.
- Commit pushed.

### Non-goals

- No new robot feature.
- No new architecture.
- No v0.5.0 capability work until documentation source-of-truth exists.

## v0.5.0 - Configurable Presets and Stricter Gates

### Objective

Turn the current matrix/sweep functionality into a cleaner experiment system with configurable presets and stronger evaluation gates.

### Motivation

v0.4.0 added config sweeps. The next bottleneck is making matrix and sweep configurations easier to reuse, compare, and evaluate rigorously.

### Outcomes

v0.5.0 should make it possible to define and run named experiment presets without manually constructing long CLI commands each time.

### Candidate capabilities

- Configurable matrix presets.
- Configurable sweep presets.
- Named gate profiles.
- Stricter gate thresholds.
- Better summary ranking logic.
- Clear failure/gate explanations in reports.
- Better validation for malformed configs.
- CLI support for listing available presets.

### Expected artifacts

- preset config files
- matrix preset reports
- sweep preset reports
- updated JSON summaries
- updated HTML reports
- updated CSV outputs
- tests for preset loading and gate evaluation

### Exit criteria

- Default preset system works.
- Users can run named matrix/sweep presets.
- Gate profiles are documented.
- Existing commands remain backwards-compatible.
- Tests pass.
- README updated.
- v0.5.0 tag and release created.

### Non-goals

- No new robot.
- No new task.
- No web UI.
- No database.
- No cloud.
- No broad benchmark suite.

## v0.6.0 - Reproducible Results Capsule

### Objective

Publish a small, honest public proof object showing that kineForge can train a policy, evaluate it across configured failures, and write inspectable local artifacts.

### Motivation

After v0.5.0 added presets and gate profiles, the strongest next public milestone was not more internal plumbing. It was a reproducible example result that shows the current system working end to end.

### Delivered capabilities

- `RESULTS.md` with exact reproduction commands.
- Curated example matrix output under `examples/results/v0.6.0/`.
- Committed matrix summary, CSV, static HTML report, scorecards, replay index, and selected trajectory PNGs.
- Clear limitations on one robot/task, placeholder collision metrics, and non-modeled friction placeholders.

### Exit criteria

- Result capsule exists.
- Reproduction commands are documented.
- Example artifacts are committed without a policy file or large generated run directory.
- Tests pass.
- v0.6.0 tag and release created.

### Non-goals

- No claim of real-world safety certification.
- No hardware deployment.
- No weapons or fire-control framing.
- No broad safety platform.


## v0.6.x / v0.8.0 - Contact, Collision, and Safety-Relevant Metrics

Contact/collision metrics remain important, but they should only be implemented when the environment can support them honestly. Do not replace the placeholder collision value with another placeholder.

## v0.7.0 - Replay and Visual Inspection Improvements

### Objective

Make evaluation results easier to inspect visually.

### Motivation

kineForge already writes PNG plots and static reports. The next step is better replay/inspection artifacts so failures are easier to understand.

### Delivered capabilities

- `replay_gallery.py` CLI for building a static gallery from `matrix_summary.json` and `replay_index.json`.
- `kineforge.gallery` helpers for deterministic gallery payloads and HTML output.
- Committed `examples/results/v0.6.0/replay_gallery.html` artifact.
- Tests for ranked scenario gallery generation and relative image links.

### Expected artifacts

- static replay gallery HTML
- existing PNG trajectory plots
- scorecard and trajectory links per ranked scenario
- tests for artifact generation

### Exit criteria

- Replay artifacts are easier to inspect.
- Existing PNG outputs remain available.
- Gallery output is dependency-light static HTML.
- Gallery links to relevant scorecards and trajectory plots.
- v0.7.0 tag and release created.

### Non-goals

- No heavy visualization framework.
- No web server.
- No interactive dashboard.
- No dependency-heavy rendering pipeline unless explicitly justified.

## v0.8.0 - Carefully Broader Tasks and Variants

### Objective

Expand beyond the single baseline task only if the existing evaluation infrastructure is strong enough.

### Motivation

After configs, gates, metrics, and replay improve, kineForge can start demonstrating breadth without becoming a robot zoo.

### Candidate capabilities

- Additional task variant using the same robot.
- Additional target distributions.
- More structured domain randomization presets.
- More meaningful failure combinations.
- Optional second simple robot embodiment only if it does not bloat architecture.

### Expected artifacts

- new task configs
- new randomization configs
- new matrix presets
- comparison reports across variants
- tests for config/task loading

### Exit criteria

- At least one meaningful new task or task variant exists.
- The architecture remains simple.
- The README still explains the project in 10 seconds.
- Tests pass.
- v0.8.0 tag and release created.

### Non-goals

- No robot zoo.
- No humanoid simulator.
- No complex manipulation suite.
- No broad benchmark claims.

## v0.9.0 - Experiment Registry and Reproducibility Polish

### Objective

Make accumulated experiment outputs easier to track, compare, and reproduce.

### Motivation

As matrix runs and sweeps accumulate, the project needs a cleaner local experiment index.

### Candidate capabilities

- Local experiment registry.
- Run index over `runs/`.
- Search/list command for past runs.
- Compare command improvements.
- Better metadata normalization.
- Better report linking.
- Optional export bundle for sharing results.
- CI workflow if appropriate.

### Expected artifacts

- run index JSON or CSV
- experiment registry output
- improved comparison reports
- documentation for reproducibility workflow

### Exit criteria

- A user can list, inspect, and compare previous runs more easily.
- Reproducibility story is clear.
- Tests pass.
- v0.9.0 tag and release created.

### Non-goals

- No database unless local files become clearly insufficient.
- No cloud experiment tracker.
- No SaaS-style productization.

## v1.0.0 - Stable Local Robot Policy Testbed

### Objective

Publish the first stable release of kineForge as a small, serious, reproducible robot policy evaluation testbed.

### v1.0.0 should include

- reliable install path
- clear quickstart
- stable CLI commands
- reproducible training
- deterministic evaluation
- scorecards
- eval matrices
- config sweeps
- meaningful gates
- contact/collision metrics where supported
- replay/inspection artifacts
- static reports
- strong README
- project doctrine
- roadmap
- passing tests
- clear limitations

### Exit criteria

- New user can clone, install, train, eval, run matrix, run sweep, and inspect artifacts.
- README is accurate.
- Tests pass.
- All core commands work.
- Example outputs are documented.
- Limitations are honest.
- Repository is clean.
- Human review completed before release.
- v1.0.0 tag and GitHub Release created manually or with explicit approval.

### Non-goals

- No claim of real robot deployment.
- No production robotics stack.
- No cloud platform.
- No broad benchmark dominance claims.
- No unnecessary UI.

## Agent Instructions for Roadmap Use

When an agent is asked to work autonomously toward a target version:

1. Read this roadmap.
2. Identify the target version.
3. Identify the current released version.
4. Work through meaningful intermediate improvements if needed.
5. Commit and push after each verified improvement.
6. For minor versions, tag and release only after all exit criteria are met.
7. Do not invent unrelated milestones.
8. Do not skip exit criteria.
9. Stop when the target milestone is complete or a defined stop condition is reached.

## Immediate Next Step

The next immediate repository task after v0.7.0 is:

```text
v0.8.0 honest physical/contact metric feasibility
```

This means inspecting whether contact/collision metrics can be implemented honestly in the current MuJoCo tabletop reach environment before changing reported safety-relevant metrics.
