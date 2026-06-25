from __future__ import annotations

from pathlib import Path
from typing import Any, Collection

from kineforge.config import config_path, load_env_configs, project_root
from kineforge.evals import run_evaluation
from kineforge.replay import (
    save_distance_over_time_png,
    save_episode_rewards_png,
    save_trajectory_png,
)
from kineforge.reports import package_versions, write_config_snapshot, write_json


def write_eval_artifacts(
    output_dir: Path,
    policy_path: Path,
    source_policy_path: Path,
    robot: str,
    task: str,
    reward: str,
    failures: Collection[str],
    episodes: int,
    seed: int,
    run_timestamp: str,
) -> dict[str, Any]:
    active_failures = set(failures)
    scorecard, replay_payload = run_evaluation(
        policy_path,
        robot,
        task,
        reward,
        active_failures,
        episodes,
        seed,
    )
    robot_config, task_config, reward_config, failure_config = load_env_configs(
        robot,
        task,
        reward,
        "basic_failures",
    )
    configs: dict[str, Any] = {
        "robot": robot_config,
        "task": task_config,
        "reward": reward_config,
    }
    if active_failures:
        configs["failures"] = failure_config
    config_snapshot_path = write_config_snapshot(output_dir, configs)

    scorecard_path = output_dir / "scorecard.json"
    trajectory_path = output_dir / "trajectory.png"
    distance_path = output_dir / "distance_over_time.png"
    rewards_path = output_dir / "episode_rewards.png"
    eval_metadata_path = output_dir / "eval_metadata.json"
    artifacts = {
        "policy_zip": str(policy_path),
        "scorecard_json": str(scorecard_path),
        "trajectory_png": str(trajectory_path),
        "distance_over_time_png": str(distance_path),
        "episode_rewards_png": str(rewards_path),
        "eval_metadata_json": str(eval_metadata_path),
        "config_snapshot_yaml": str(config_snapshot_path),
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
    config_paths = {
        "robot": str(config_path("robots", robot).relative_to(root)),
        "task": str(config_path("tasks", task).relative_to(root)),
        "reward": str(config_path("rewards", reward).relative_to(root)),
    }
    if active_failures:
        config_paths["failures"] = str(config_path("failures", "basic_failures").relative_to(root))
    write_json(
        eval_metadata_path,
        {
            "timestamp": run_timestamp,
            "policy_path": str(policy_path),
            "source_policy_path": str(source_policy_path),
            "seed": seed,
            "failure_modes": sorted(active_failures),
            "episodes": episodes,
            "gate_thresholds": scorecard["gate"]["thresholds"],
            "config_paths": config_paths,
            "config_snapshot_path": str(config_snapshot_path),
            "artifacts": artifacts,
            "versions": package_versions(),
        },
    )
    return {
        "scorecard": scorecard,
        "scorecard_path": scorecard_path,
        "artifacts": artifacts,
        "metadata_path": eval_metadata_path,
    }
