from __future__ import annotations

import argparse
from pathlib import Path

from kineforge.gates import list_gate_profiles
from kineforge.eval_artifacts import write_eval_artifacts
from kineforge.matrix import parse_failures
from kineforge.reports import copy_file, prepare_run_dir, reset_latest_dir, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a kineForge PPO policy.")
    parser.add_argument("--policy", default="runs/latest/policy.zip")
    parser.add_argument("--failures", default="")
    parser.add_argument("--task", default="tabletop_reach")
    parser.add_argument("--robot", default="arm_v0")
    parser.add_argument("--reward", default="reach_v0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate", default="standard", help="Gate profile name from configs/gates.")
    parser.add_argument("--list-gates", action="store_true", help="List gate profiles, then exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_gates:
        print("Gate profiles:")
        for profile in list_gate_profiles():
            print(f"  {profile}")
        return
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

    result = write_eval_artifacts(
        output_dir=eval_dir,
        policy_path=policy_snapshot_path,
        source_policy_path=source_policy_path,
        robot=args.robot,
        task=args.task,
        reward=args.reward,
        failures=active_failures,
        episodes=args.episodes,
        seed=args.seed,
        run_timestamp=run_timestamp,
        gate_profile=args.gate,
    )
    scorecard = result["scorecard"]
    scorecard_path = result["scorecard_path"]
    eval_metadata_path = result["metadata_path"]
    artifacts = result["artifacts"]
    trajectory_path = artifacts["trajectory_png"]
    distance_path = artifacts["distance_over_time_png"]
    rewards_path = artifacts["episode_rewards_png"]

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
    print(f"Gate profile: {scorecard['gate']['profile']}")
    print(f"Gate explanation: {scorecard['gate']['explanation']}")
    print(f"Scorecard: {scorecard_path}")
    print(f"Eval metadata: {eval_metadata_path}")
    print(f"Trajectory PNG: {trajectory_path}")
    print(f"Distance PNG: {distance_path}")
    print(f"Episode rewards PNG: {rewards_path}")


if __name__ == "__main__":
    main()
