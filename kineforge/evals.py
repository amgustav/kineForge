from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from kineforge.config import load_env_configs
from kineforge.gates import DEFAULT_GATE_PROFILE, GateProfile, build_gate_result, load_gate_profile
from kineforge.envs import TabletopReachEnv


def build_scorecard(
    policy_path: Path,
    robot: str,
    task: str,
    reward: str,
    seed: int,
    failure_modes: Collection[str],
    episode_results: list[dict[str, Any]],
    success_threshold: float,
    gate_profile: GateProfile | str | None = None,
) -> dict[str, Any]:
    success_rate = float(np.mean([bool(result["success"]) for result in episode_results]))
    mean_final_distance = float(np.mean([float(result["final_distance"]) for result in episode_results]))
    timeout_rate = float(np.mean([bool(result["timeout"]) for result in episode_results]))
    mean_episode_reward = float(np.mean([float(result["episode_reward"]) for result in episode_results]))
    collision_rate = 0.0

    if gate_profile is None:
        profile = load_gate_profile(DEFAULT_GATE_PROFILE)
    elif isinstance(gate_profile, str):
        profile = load_gate_profile(gate_profile)
    else:
        profile = gate_profile
    gate = build_gate_result(
        profile=profile,
        success_threshold=success_threshold,
        success_rate=success_rate,
        mean_final_distance=mean_final_distance,
        timeout_rate=timeout_rate,
    )

    return {
        "policy_path": str(policy_path),
        "task": task,
        "robot": robot,
        "reward": reward,
        "seed": int(seed),
        "episodes": len(episode_results),
        "failure_modes": sorted(failure_modes),
        "summary": {
            "success_rate": success_rate,
            "mean_final_distance": mean_final_distance,
            "timeout_rate": timeout_rate,
            "mean_episode_reward": mean_episode_reward,
            "collision_rate": collision_rate,
        },
        "gate": gate,
        "per_episode": episode_results,
        "collision_rate_explanation": (
            "Collision detection is not implemented in v0.2.0; collision_rate is reported as 0.0 "
            "and should not be interpreted as a safety metric."
        ),
    }


def run_evaluation(
    policy_path: Path,
    robot: str,
    task: str,
    reward: str,
    failures: Collection[str],
    episodes: int,
    seed: int,
    gate_profile: GateProfile | str | None = None,
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
    success_threshold = float(task_config["success_threshold"])
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

    episode_results: list[dict[str, Any]] = []
    episode_rewards: list[float] = []
    first_episode_replay: dict[str, Any] | None = None

    for episode_index in range(episodes):
        episode_seed = seed + episode_index
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
        episode_reward = float(total_reward)
        episode_rewards.append(episode_reward)

        episode_results.append(
            {
                "episode": int(episode_index),
                "seed": int(episode_seed),
                "success": success,
                "timeout": timeout,
                "final_distance": final_distance,
                "episode_reward": episode_reward,
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
        robot=robot,
        task=task,
        reward=reward,
        seed=seed,
        failure_modes=active_failures,
        episode_results=episode_results,
        success_threshold=success_threshold,
        gate_profile=gate_profile,
    )
    env.close()
    assert first_episode_replay is not None
    replay_payload = {
        **first_episode_replay,
        "episode_rewards": episode_rewards,
        "success_threshold": success_threshold,
    }
    return scorecard, replay_payload
