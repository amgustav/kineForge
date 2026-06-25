from __future__ import annotations

import argparse
from pathlib import Path

from kineforge.evals import run_evaluation
from kineforge.replay import save_trajectory_png
from kineforge.reports import copy_file, ensure_latest_dir, prepare_run_dir, write_json


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
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = prepare_run_dir("eval")
    latest_dir = ensure_latest_dir()
    scorecard, replay_payload = run_evaluation(
        Path(args.policy),
        args.robot,
        args.task,
        args.reward,
        parse_failures(args.failures),
        args.episodes,
        args.seed,
    )

    scorecard_path = eval_dir / "scorecard.json"
    trajectory_path = eval_dir / "trajectory.png"
    write_json(scorecard_path, scorecard)
    save_trajectory_png(
        replay_payload["trajectory"],
        replay_payload["target_position"],
        replay_payload["final_position"],
        replay_payload["success"],
        trajectory_path,
    )

    latest_scorecard = latest_dir / "scorecard.json"
    latest_trajectory = latest_dir / "trajectory.png"
    copy_file(scorecard_path, latest_scorecard)
    copy_file(trajectory_path, latest_trajectory)

    print(f"Gate status: {scorecard['gate']['status']}")
    print(f"Scorecard: {scorecard_path}")
    print(f"Trajectory PNG: {trajectory_path}")


if __name__ == "__main__":
    main()
