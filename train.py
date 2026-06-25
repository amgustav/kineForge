from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from kineforge.config import load_env_configs, write_config_snapshot
from kineforge.envs import TabletopReachEnv
from kineforge.reports import prepare_run_dir, reset_latest_dir, timestamp, write_json


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
    train_dir = prepare_run_dir("train")
    latest_dir = reset_latest_dir()

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
    latest_policy = latest_dir / "policy.zip"
    model.save(str(timestamped_policy))
    model.save(str(latest_policy))

    configs = {
        "robot": robot_config,
        "task": task_config,
        "reward": reward_config,
        "failures": failure_config,
    }
    write_config_snapshot(train_dir, configs)
    write_config_snapshot(latest_dir, configs)

    for metadata_path, policy_path in (
        (train_dir / "training_metadata.json", timestamped_policy),
        (latest_dir / "training_metadata.json", latest_policy),
    ):
        write_json(
            metadata_path,
            {
                "task": args.task,
                "robot": args.robot,
                "reward": args.reward,
                "timesteps": args.timesteps,
                "seed": args.seed,
                "ppo": ppo_config,
                "recommended_timesteps": task_config["training"]["recommended_timesteps"],
                "policy_path": str(policy_path),
                "timestamped_run_dir": str(train_dir),
                "timestamp": run_timestamp,
            },
        )

    env.close()
    print(f"Timestamped policy: {timestamped_policy}")
    print(f"Latest policy: {latest_policy}")


if __name__ == "__main__":
    main()
