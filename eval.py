from __future__ import annotations

import argparse
from pathlib import Path

from kineforge.config import config_path, load_env_configs, project_root
from kineforge.evals import run_evaluation
from kineforge.replay import (
    save_distance_over_time_png,
    save_episode_rewards_png,
    save_trajectory_png,
)
from kineforge.reports import (
    copy_file,
    package_versions,
    prepare_run_dir,
    reset_latest_dir,
    timestamp,
    write_config_snapshot,
    write_json,
)


def parse_failures(raw_failures: str) -> set[str]:
    return {failure.strip() for failure in raw_failures.split(",") if failure.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a kineForge PPO policy.")
    parser.add_argument("--policy", default="runs/latest/policy.zip")
    parser.add_argument("--failures", default="")
    parser.add_argument("--task", default="tabletop_reach")
    parser.add_argument("--robot", default="arm_v0")
    parser.add_argument("--reward", default="reach_v0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    active_failures = parse_failures(args.failures)
    run_timestamp = timestamp()
    eval_dir = prepare_run_dir("eval", run_timestamp)

    source_policy_path = Path(args.policy)
    policy_snapshot_path = eval_dir / "policy.zip"
    copy_file(source_policy_path, policy_snapshot_path)

    source_train_metadata = source_policy_path.parent / "train_metadata.json"
    eval_train_metadata = eval_dir / "train_metadata.json"
    if source_train_metadata.exists():
        copy_file(source_train_metadata, eval_train_metadata)

    scorecard, replay_payload = run_evaluation(
        policy_snapshot_path,
        args.robot,
        args.task,
        args.reward,
        active_failures,
        args.episodes,
        args.seed,
    )
    robot_config, task_config, reward_config, failure_config = load_env_configs(
        args.robot,
        args.task,
        args.reward,
        "basic_failures",
    )
    configs = {
        "robot": robot_config,
        "task": task_config,
        "reward": reward_config,
    }
    if active_failures:
        configs["failures"] = failure_config
    config_snapshot_path = write_config_snapshot(eval_dir, configs)

    scorecard_path = eval_dir / "scorecard.json"
    trajectory_path = eval_dir / "trajectory.png"
    distance_path = eval_dir / "distance_over_time.png"
    rewards_path = eval_dir / "episode_rewards.png"
    eval_metadata_path = eval_dir / "eval_metadata.json"
    artifacts = {
        "policy_zip": str(policy_snapshot_path),
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
        "robot": str(config_path("robots", args.robot).relative_to(root)),
        "task": str(config_path("tasks", args.task).relative_to(root)),
        "reward": str(config_path("rewards", args.reward).relative_to(root)),
    }
    if active_failures:
        config_paths["failures"] = str(config_path("failures", "basic_failures").relative_to(root))
    write_json(
        eval_metadata_path,
        {
            "timestamp": run_timestamp,
            "policy_path": str(policy_snapshot_path),
            "source_policy_path": str(source_policy_path),
            "seed": args.seed,
            "failure_modes": sorted(active_failures),
            "episodes": args.episodes,
            "gate_thresholds": scorecard["gate"]["thresholds"],
            "config_paths": config_paths,
            "config_snapshot_path": str(config_snapshot_path),
            "artifacts": artifacts,
            "versions": package_versions(),
        },
    )

    latest_dir = reset_latest_dir()
    for filename in (
        "policy.zip",
        "scorecard.json",
        "eval_metadata.json",
        "config_snapshot.yaml",
        "trajectory.png",
        "distance_over_time.png",
        "episode_rewards.png",
    ):
        copy_file(eval_dir / filename, latest_dir / filename)
    if eval_train_metadata.exists():
        copy_file(eval_train_metadata, latest_dir / "train_metadata.json")

    print(f"Gate status: {scorecard['gate']['status']}")
    print(f"Scorecard: {scorecard_path}")
    print(f"Eval metadata: {eval_metadata_path}")
    print(f"Trajectory PNG: {trajectory_path}")
    print(f"Distance PNG: {distance_path}")
    print(f"Episode rewards PNG: {rewards_path}")


if __name__ == "__main__":
    main()
