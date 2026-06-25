from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from kineforge.config import load_env_configs
from kineforge.envs import TabletopReachEnv


def run_evaluation(
    policy_path: Path,
    robot: str,
    task: str,
    reward: str,
    failures: Collection[str],
    episodes: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    robot_config, task_config, reward_config, failure_config = load_env_configs(
        robot,
        task,
        reward,
        "basic_failures",
    )
    active_failures = set(failures)
    env = TabletopReachEnv(
        robot_config,
        task_config,
        reward_config,
        failure_config,
        active_failures=active_failures,
        training=False,
        seed=seed,
    )
    model = PPO.load(str(policy_path), env=env)

    successes: list[bool] = []
    final_distances: list[float] = []
    timeouts: list[bool] = []
    episode_rewards: list[float] = []
    replay_payload: dict[str, Any] | None = None

    for episode_index in range(episodes):
        obs, info = env.reset(seed=seed + episode_index)
        terminated = False
        truncated = False
        total_reward = 0.0
        final_info = info

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward_value, terminated, truncated, final_info = env.step(action)
            total_reward += float(reward_value)

        success = bool(final_info["success"])
        timeout = bool(final_info["timeout"])
        final_distance = float(final_info["distance"])
        successes.append(success)
        timeouts.append(timeout)
        final_distances.append(final_distance)
        episode_rewards.append(total_reward)

        if replay_payload is None:
            trajectory = np.asarray(env.trajectory, dtype=np.float64)
            replay_payload = {
                "trajectory": trajectory,
                "target_position": np.asarray(final_info["target_position"], dtype=np.float64),
                "final_position": trajectory[-1].copy(),
                "success": success,
            }

    success_rate = float(np.mean(successes))
    mean_final_distance = float(np.mean(final_distances))
    timeout_rate = float(np.mean(timeouts))
    mean_episode_reward = float(np.mean(episode_rewards))

    criteria = {
        "success_rate >= 0.80": bool(success_rate >= 0.80),
        f"mean_final_distance <= {task_config['success_threshold']}": bool(
            mean_final_distance <= float(task_config["success_threshold"])
        ),
        "timeout_rate <= 0.30": bool(timeout_rate <= 0.30),
    }
    failed_criteria = [name for name, passed in criteria.items() if not passed]

    scorecard = {
        "policy_path": str(policy_path),
        "task": task,
        "robot": robot,
        "reward": reward,
        "failures": sorted(active_failures),
        "episodes": int(episodes),
        "success_rate": success_rate,
        "mean_final_distance": mean_final_distance,
        "timeout_rate": timeout_rate,
        "mean_episode_reward": mean_episode_reward,
        "collision_rate": 0.0,
        "collision_note": "Collision detection is not implemented in v0; collision_rate is fixed at 0.0.",
        "gate": {
            "status": "PASS" if not failed_criteria else "FAIL",
            "criteria": criteria,
            "failed_criteria": failed_criteria,
        },
    }
    env.close()
    assert replay_payload is not None
    return scorecard, replay_payload
