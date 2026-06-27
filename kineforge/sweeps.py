from __future__ import annotations

import csv
import html
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from stable_baselines3 import PPO

from kineforge.config import config_path, load_env_configs, load_yaml, project_root
from kineforge.envs import TabletopReachEnv
from kineforge.evals import build_scorecard
from kineforge.gates import DEFAULT_GATE_PROFILE, load_gate_profile
from kineforge.replay import (
    save_distance_over_time_png,
    save_episode_rewards_png,
    save_trajectory_png,
)
from kineforge.reports import package_versions, write_config_snapshot, write_json

VARIANT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
SUMMARY_FIELDS = (
    "success_rate",
    "mean_final_distance",
    "timeout_rate",
    "mean_episode_reward",
    "collision_rate",
)


@dataclass(frozen=True)
class SweepVariant:
    name: str
    robot: str
    task: str
    reward: str
    failures: tuple[str, ...]
    episodes: int
    seed: int
    timesteps: int
    description: str
    gate_profile: str
    task_overrides: dict[str, Any]
    reward_overrides: dict[str, Any]


@dataclass(frozen=True)
class SweepConfig:
    name: str
    path: Path
    gate_profile: str
    description: str
    variants: tuple[SweepVariant, ...]


def merge_config(base: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    if not overrides:
        return merged
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _as_positive_int(value: Any, field_name: str) -> int:
    int_value = int(value)
    if int_value < 1:
        raise ValueError(f"{field_name} must be at least 1")
    return int_value


def _as_failures(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        failures = [item.strip() for item in value.split(",")]
    else:
        failures = [str(item).strip() for item in value]
    return tuple(sorted(item for item in failures if item))


def _validate_variant_name(name: str) -> str:
    if not name:
        raise ValueError("variant name must not be empty")
    if not VARIANT_NAME_PATTERN.match(name):
        raise ValueError(f"invalid variant name: {name!r}")
    return name


def list_sweep_presets() -> list[str]:
    sweep_dir = project_root() / "configs" / "sweeps"
    return sorted(path.stem for path in sweep_dir.glob("*.yaml") if path.is_file())


def load_sweep_config(
    path: str | Path,
    *,
    seed_override: int | None = None,
    timesteps_override: int | None = None,
    episodes_override: int | None = None,
    gate_override: str | None = None,
) -> SweepConfig:
    config_path_value = Path(path)
    payload = load_yaml(config_path_value)
    if not isinstance(payload, Mapping):
        raise ValueError("sweep config must be a mapping")
    sweep_name = str(payload.get("name", config_path_value.stem)).strip() or config_path_value.stem
    sweep_description = str(payload.get("description", ""))
    sweep_gate_profile = str(gate_override or payload.get("gate_profile", DEFAULT_GATE_PROFILE))
    raw_variants = payload.get("variants", [])
    if not isinstance(raw_variants, list):
        raise ValueError("sweep config variants must be a list")
    if len(raw_variants) < 2:
        raise ValueError("sweep config must define at least two variants")

    default_robot = str(payload.get("robot", "arm_v0"))
    default_task = str(payload.get("task", "tabletop_reach"))
    default_reward = str(payload.get("reward", "reach_v0"))
    default_failures = _as_failures(payload.get("failures", ()))
    default_episodes = _as_positive_int(payload.get("episodes", 10), "episodes")
    default_seed = payload.get("seed")
    default_timesteps = payload.get("timesteps")

    variants: list[SweepVariant] = []
    names: list[str] = []
    for raw_variant in raw_variants:
        if not isinstance(raw_variant, Mapping):
            raise ValueError("sweep variant entries must be mappings")
        name = _validate_variant_name(str(raw_variant["name"]).strip())
        names.append(name)
        seed_source = seed_override if seed_override is not None else raw_variant.get("seed", default_seed)
        timesteps_source = (
            timesteps_override if timesteps_override is not None else raw_variant.get("timesteps", default_timesteps)
        )
        if seed_source is None:
            raise ValueError(f"variant {name!r} must define seed or use --seed")
        if timesteps_source is None:
            raise ValueError(f"variant {name!r} must define timesteps or use --timesteps")
        episodes_source = episodes_override if episodes_override is not None else raw_variant.get("episodes", default_episodes)
        variants.append(
            SweepVariant(
                name=name,
                robot=str(raw_variant.get("robot", default_robot)),
                task=str(raw_variant.get("task", default_task)),
                reward=str(raw_variant.get("reward", default_reward)),
                failures=_as_failures(raw_variant.get("failures", default_failures)),
                episodes=_as_positive_int(episodes_source, f"variant {name} episodes"),
                seed=int(seed_source),
                timesteps=_as_positive_int(timesteps_source, f"variant {name} timesteps"),
                description=str(raw_variant.get("description", "")),
                gate_profile=str(gate_override or raw_variant.get("gate_profile", sweep_gate_profile)),
                task_overrides=dict(raw_variant.get("task_overrides", {})),
                reward_overrides=dict(raw_variant.get("reward_overrides", {})),
            )
        )
    if len(set(names)) != len(names):
        raise ValueError("sweep config contains duplicate variant names")
    load_gate_profile(sweep_gate_profile)
    for variant in variants:
        load_gate_profile(variant.gate_profile)
    return SweepConfig(
        name=sweep_name,
        path=config_path_value,
        gate_profile=sweep_gate_profile,
        description=sweep_description,
        variants=tuple(variants),
    )


def _evaluate_policy_with_configs(
    *,
    policy_path: Path,
    variant: SweepVariant,
    robot_config: Mapping[str, Any],
    task_config: Mapping[str, Any],
    reward_config: Mapping[str, Any],
    failure_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    active_failures = set(variant.failures)
    env = TabletopReachEnv(
        dict(robot_config),
        dict(task_config),
        dict(reward_config),
        dict(failure_config),
        active_failures=active_failures,
        training=False,
        seed=variant.seed,
    )
    model = PPO.load(str(policy_path), env=env)
    episode_results: list[dict[str, Any]] = []
    episode_rewards: list[float] = []
    first_episode_replay: dict[str, Any] | None = None

    for episode_index in range(variant.episodes):
        episode_seed = variant.seed + episode_index
        obs, info = env.reset(seed=episode_seed)
        distance_history = [float(info["distance"])]
        terminated = False
        truncated = False
        total_reward = 0.0
        final_info = info
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward_value, terminated, truncated, final_info = env.step(action)
            total_reward += float(reward_value)
            distance_history.append(float(final_info["distance"]))

        success = bool(final_info["success"])
        timeout = bool(final_info["timeout"])
        final_distance = float(final_info["distance"])
        episode_rewards.append(float(total_reward))
        episode_results.append(
            {
                "episode": int(episode_index),
                "seed": int(episode_seed),
                "success": success,
                "timeout": timeout,
                "final_distance": final_distance,
                "episode_reward": float(total_reward),
                "steps": int(env.step_count),
                "target_position": np.asarray(final_info["target_position"], dtype=np.float64).tolist(),
                "final_position": np.asarray(final_info["end_effector_position"], dtype=np.float64).tolist(),
                "active_failures": list(final_info["active_failures"]),
            }
        )
        if first_episode_replay is None:
            first_episode_replay = {
                "trajectory": np.asarray(env.trajectory, dtype=np.float64),
                "target_position": np.asarray(final_info["target_position"], dtype=np.float64),
                "final_position": np.asarray(final_info["end_effector_position"], dtype=np.float64),
                "success": success,
                "distance_history": distance_history,
            }

    scorecard = build_scorecard(
        policy_path=policy_path,
        robot=variant.robot,
        task=variant.task,
        reward=variant.reward,
        seed=variant.seed,
        failure_modes=active_failures,
        episode_results=episode_results,
        success_threshold=float(task_config["success_threshold"]),
        gate_profile=variant.gate_profile,
    )
    scorecard["variant"] = variant.name
    env.close()
    assert first_episode_replay is not None
    return scorecard, {
        **first_episode_replay,
        "episode_rewards": episode_rewards,
        "success_threshold": float(task_config["success_threshold"]),
    }


def run_sweep_variant(output_dir: Path, sweep_config: SweepConfig, variant: SweepVariant, run_timestamp: str) -> dict[str, Any]:
    robot_config, base_task_config, base_reward_config, failure_config = load_env_configs(
        variant.robot,
        variant.task,
        variant.reward,
        "basic_failures",
    )
    task_config = merge_config(base_task_config, variant.task_overrides)
    reward_config = merge_config(base_reward_config, variant.reward_overrides)
    ppo_config = dict(task_config["training"]["ppo"])

    output_dir.mkdir(parents=True, exist_ok=True)
    train_env = TabletopReachEnv(
        robot_config,
        task_config,
        reward_config,
        failure_config,
        active_failures=set(),
        training=True,
        seed=variant.seed,
    )
    model = PPO("MlpPolicy", train_env, verbose=0, seed=variant.seed, **ppo_config)
    model.learn(total_timesteps=variant.timesteps)
    policy_path = output_dir / "policy.zip"
    model.save(str(policy_path))
    train_env.close()

    config_snapshot_path = write_config_snapshot(
        output_dir,
        {
            "sweep": {"name": sweep_config.name, "config_path": str(sweep_config.path)},
            "variant": {
                "name": variant.name,
                "description": variant.description,
                "seed": variant.seed,
                "timesteps": variant.timesteps,
                "episodes": variant.episodes,
                "failures": list(variant.failures),
                "gate_profile": variant.gate_profile,
                "task_overrides": variant.task_overrides,
                "reward_overrides": variant.reward_overrides,
            },
            "robot": robot_config,
            "task": task_config,
            "reward": reward_config,
            "failures": failure_config,
            "ppo": ppo_config,
        },
    )

    scorecard, replay_payload = _evaluate_policy_with_configs(
        policy_path=policy_path,
        variant=variant,
        robot_config=robot_config,
        task_config=task_config,
        reward_config=reward_config,
        failure_config=failure_config,
    )
    scorecard_path = output_dir / "scorecard.json"
    eval_metadata_path = output_dir / "eval_metadata.json"
    trajectory_path = output_dir / "trajectory.png"
    distance_path = output_dir / "distance_over_time.png"
    rewards_path = output_dir / "episode_rewards.png"
    artifacts = {
        "policy_zip": str(policy_path),
        "scorecard_json": str(scorecard_path),
        "eval_metadata_json": str(eval_metadata_path),
        "config_snapshot_yaml": str(config_snapshot_path),
        "trajectory_png": str(trajectory_path),
        "distance_over_time_png": str(distance_path),
        "episode_rewards_png": str(rewards_path),
    }
    scorecard["artifacts"] = artifacts
    write_json(scorecard_path, scorecard)
    save_trajectory_png(
        replay_payload["trajectory"],
        replay_payload["target_position"],
        replay_payload["final_position"],
        replay_payload["success"],
        trajectory_path,
    )
    save_distance_over_time_png(
        replay_payload["distance_history"],
        replay_payload["success_threshold"],
        distance_path,
    )
    save_episode_rewards_png(replay_payload["episode_rewards"], rewards_path)

    root = project_root()
    write_json(
        eval_metadata_path,
        {
            "timestamp": run_timestamp,
            "sweep": sweep_config.name,
            "variant": variant.name,
            "policy_path": str(policy_path),
            "seed": variant.seed,
            "timesteps": variant.timesteps,
            "episodes": variant.episodes,
            "failure_modes": list(variant.failures),
            "gate_thresholds": scorecard["gate"]["thresholds"],
            "gate_profile": scorecard["gate"]["profile"],
            "gate_explanation": scorecard["gate"]["explanation"],
            "config_paths": {
                "sweep": str(sweep_config.path),
                "robot": str(config_path("robots", variant.robot).relative_to(root)),
                "task": str(config_path("tasks", variant.task).relative_to(root)),
                "reward": str(config_path("rewards", variant.reward).relative_to(root)),
                "failures": str(config_path("failures", "basic_failures").relative_to(root)),
            },
            "config_snapshot_path": str(config_snapshot_path),
            "artifacts": artifacts,
            "versions": package_versions(),
        },
    )
    return {
        "variant": variant,
        "scorecard": scorecard,
        "scorecard_path": scorecard_path,
        "metadata_path": eval_metadata_path,
        "artifacts": artifacts,
    }


def _rank_key(row: Mapping[str, Any]) -> tuple[int, float, float, str]:
    gate_rank = 0 if row["gate_status"] == "PASS" else 1
    return (gate_rank, -float(row["success_rate"]), float(row["mean_final_distance"]), str(row["variant"]))


def rank_variant_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked_rows = [dict(row) for row in sorted(rows, key=_rank_key)]
    for index, row in enumerate(ranked_rows, start=1):
        row["rank"] = index
    return ranked_rows


def build_sweep_summary(
    *,
    sweep_config: SweepConfig,
    output_dir: Path,
    run_timestamp: str,
    variant_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, result in variant_results.items():
        variant = result["variant"]
        scorecard = result["scorecard"]
        summary = scorecard["summary"]
        row = {
            "variant": name,
            "description": variant.description,
            "gate_status": scorecard["gate"]["status"],
            "gate_profile": scorecard["gate"]["profile"],
            "gate_failed_criteria": list(scorecard["gate"].get("failed_criteria", ())),
            "gate_explanation": scorecard["gate"].get("explanation", ""),
            "physical_metrics": scorecard.get("physical_metrics", {}),
            "failure_mode_metadata": scorecard.get("failure_mode_metadata", {}),
            "seed": variant.seed,
            "timesteps": variant.timesteps,
            "episodes": variant.episodes,
            "failures": list(variant.failures),
            "policy_zip": result["artifacts"]["policy_zip"],
            "scorecard_json": str(result["scorecard_path"]),
            "eval_metadata_json": str(result["metadata_path"]),
            "config_snapshot_yaml": result["artifacts"]["config_snapshot_yaml"],
        }
        for field in SUMMARY_FIELDS:
            row[field] = summary.get(field, "n/a")
        rows.append(row)

    ranked_rows = rank_variant_rows(rows)
    return {
        "timestamp": run_timestamp,
        "run_id": f"sweep-{run_timestamp}",
        "sweep": sweep_config.name,
        "config_path": str(sweep_config.path),
        "output_dir": str(output_dir),
        "description": sweep_config.description,
        "gate_profile": sweep_config.gate_profile,
        "variant_count": len(ranked_rows),
        "gate": {
            "pass_count": sum(1 for row in ranked_rows if row["gate_status"] == "PASS"),
            "fail_count": sum(1 for row in ranked_rows if row["gate_status"] != "PASS"),
        },
        "ranking": ["gate_status", "success_rate desc", "mean_final_distance asc"],
        "ranked_variants": ranked_rows,
        "variants": {row["variant"]: row for row in ranked_rows},
    }


def write_sweep_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "rank",
        "variant",
        "description",
        "gate_profile",
        "gate_status",
        "failed_criteria",
        "success_rate",
        "mean_final_distance",
        "timeout_rate",
        "mean_episode_reward",
        "collision_rate",
        "seed",
        "timesteps",
        "episodes",
        "failures",
        "policy_zip",
        "scorecard_json",
        "eval_metadata_json",
        "config_snapshot_yaml",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["ranked_variants"]:
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in fieldnames},
                    "run_id": summary["run_id"],
                    "failed_criteria": ",".join(row.get("gate_failed_criteria", ())),
                    "failures": ",".join(row.get("failures", ())),
                }
            )


def write_sweep_report_html(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in summary["ranked_variants"]:
        rows.append(
            "<tr>"
            f"<td>{int(row['rank'])}</td>"
            f"<td>{html.escape(str(row['variant']))}</td>"
            f"<td>{html.escape(str(row['gate_profile']))}</td>"
            f"<td>{html.escape(str(row['gate_status']))}</td>"
            f"<td>{html.escape(', '.join(row.get('gate_failed_criteria', ())))}</td>"
            f"<td>{html.escape(str(row.get('gate_explanation', '')))}</td>"
            f"<td>{float(row['success_rate']):.3f}</td>"
            f"<td>{float(row['mean_final_distance']):.4f}</td>"
            f"<td>{float(row['timeout_rate']):.3f}</td>"
            f"<td>{int(row['seed'])}</td>"
            f"<td>{int(row['timesteps'])}</td>"
            f"<td>{html.escape(','.join(row.get('failures', ())))}</td>"
            f"<td><a href=\"{html.escape(_relative_link(row['scorecard_json'], path.parent))}\">scorecard</a></td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>kineForge sweep {html.escape(str(summary['run_id']))}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d0d7de; padding: 0.45rem 0.6rem; text-align: left; }}
    th {{ background: #f6f8fa; }}
    code {{ background: #f6f8fa; padding: 0.1rem 0.25rem; }}
  </style>
</head>
<body>
  <h1>kineForge config sweep</h1>
  <p><strong>Run:</strong> <code>{html.escape(str(summary['run_id']))}</code></p>
  <p><strong>Gate profile:</strong> <code>{html.escape(str(summary['gate_profile']))}</code></p>
  <p><strong>Ranking:</strong> gate status, success rate descending, mean final distance ascending.</p>
  <p><strong>Variants:</strong> {int(summary['variant_count'])}; PASS {int(summary['gate']['pass_count'])}, FAIL {int(summary['gate']['fail_count'])}</p>
  <table>
    <thead>
      <tr><th>Rank</th><th>Variant</th><th>Gate profile</th><th>Gate</th><th>Failed criteria</th><th>Gate explanation</th><th>Success rate</th><th>Mean final distance</th><th>Timeout rate</th><th>Seed</th><th>Timesteps</th><th>Failures</th><th>Scorecard</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _relative_link(path_value: str, output_dir: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        try:
            return str(path.relative_to(output_dir))
        except ValueError:
            return str(path)
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)
