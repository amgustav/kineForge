from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from kineforge.config import config_path, load_env_configs, project_root
from kineforge.envs import TabletopReachEnv
from kineforge.reports import (
    package_versions,
    prepare_run_dir,
    reset_latest_dir,
    timestamp,
    write_config_snapshot,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train kineForge PPO policy.")
    parser.add_argument("--task", default="tabletop_reach")
    parser.add_argument("--robot", default="arm_v0")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--reward", default="reach_v0")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot_config, task_config, reward_config, failure_config = load_env_configs(
        args.robot,
        args.task,
        args.reward,
        "basic_failures",
    )
    ppo_config = dict(task_config["training"]["ppo"])
    run_timestamp = timestamp()
    train_dir = prepare_run_dir("train", run_timestamp)

    env = TabletopReachEnv(
        robot_config,
        task_config,
        reward_config,
        failure_config,
        active_failures=set(),
        training=True,
        seed=args.seed,
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        **ppo_config,
    )
    model.learn(total_timesteps=args.timesteps)

    timestamped_policy = train_dir / "policy.zip"
    model.save(str(timestamped_policy))

    configs = {
        "robot": robot_config,
        "task": task_config,
        "reward": reward_config,
        "failures": failure_config,
        "ppo": ppo_config,
    }
    timestamped_config_snapshot = write_config_snapshot(train_dir, configs)

    root = project_root()
    config_paths = {
        "robot": str(config_path("robots", args.robot).relative_to(root)),
        "task": str(config_path("tasks", args.task).relative_to(root)),
        "reward": str(config_path("rewards", args.reward).relative_to(root)),
        "failures": str(config_path("failures", "basic_failures").relative_to(root)),
    }
    base_metadata = {
        "timestamp": run_timestamp,
        "task": args.task,
        "robot": args.robot,
        "reward": args.reward,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "recommended_timesteps": task_config["training"]["recommended_timesteps"],
        "ppo_hyperparameters": ppo_config,
        "config_paths": config_paths,
        "versions": package_versions(),
    }
    write_json(
        train_dir / "train_metadata.json",
        {
            **base_metadata,
            "policy_path": str(timestamped_policy),
            "config_snapshot_path": str(timestamped_config_snapshot),
        },
    )

    latest_dir = reset_latest_dir()
    latest_policy = latest_dir / "policy.zip"
    model.save(str(latest_policy))
    latest_config_snapshot = write_config_snapshot(latest_dir, configs)
    write_json(
        latest_dir / "train_metadata.json",
        {
            **base_metadata,
            "policy_path": str(latest_policy),
            "config_snapshot_path": str(latest_config_snapshot),
        },
    )

    env.close()
    print(f"Timestamped policy: {timestamped_policy}")
    print(f"Latest policy: {latest_policy}")


if __name__ == "__main__":
    main()
