from __future__ import annotations

from collections.abc import Collection
from typing import Any, Mapping

import gymnasium as gym
import mujoco
import numpy as np

from kineforge.randomization import (
    action_noise_std,
    apply_moved_target,
    observation_noise_std,
    sample_actuator_scale,
    sample_initial_qpos,
    sample_target_position,
    weak_actuator_scale,
)
from kineforge.rewards import compute_reach_reward


class TabletopReachEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        robot_config: Mapping[str, Any],
        task_config: Mapping[str, Any],
        reward_config: Mapping[str, Any],
        failure_config: Mapping[str, Any] | None = None,
        active_failures: Collection[str] | None = None,
        training: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        self.robot_config = robot_config
        self.task_config = task_config
        self.reward_config = reward_config
        self.failure_config = failure_config or {}
        self.active_failures = frozenset(active_failures or ())
        self.training = training

        self.model = mujoco.MjModel.from_xml_string(self._make_mjcf())
        self.data = mujoco.MjData(self.model)
        self.end_effector_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.target_mocap_id = int(self.model.body_mocapid[target_body_id])
        if self.target_mocap_id < 0:
            raise ValueError("target body must be a mocap body")

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32,
        )
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        self.step_count = 0
        self.trajectory: list[np.ndarray] = []
        self.target_position = np.asarray(self.task_config["target"]["default_position"], dtype=np.float64)
        self.actuator_scale = 1.0
        self._last_reward_terms = {
            "distance": 0.0,
            "progress": 0.0,
            "success_bonus": 0.0,
            "action_penalty": 0.0,
            "timeout_penalty": 0.0,
        }
        self.reset(seed=seed)

    def _make_mjcf(self) -> str:
        link_lengths = self.robot_config["link_lengths"]
        joint_min, joint_max = self.robot_config["joint_limit"]
        control_limit = float(self.robot_config["control_limit"])
        gear = float(self.robot_config["gear"])
        base_height = float(self.robot_config["base_height"])
        timestep = float(self.task_config["simulation"]["timestep"])
        target = self.task_config["target"]["default_position"]
        return f"""
<mujoco model="tabletop_reach">
  <option timestep="{timestep}"/>
  <worldbody>
    <geom name="tabletop" type="plane" pos="0 0 0" size="1 1 0.02" rgba="0.55 0.55 0.55 1"/>
    <body name="link1" pos="0 0 {base_height}">
      <joint name="shoulder" type="hinge" axis="0 0 1" limited="true" range="{joint_min} {joint_max}" damping="0.1"/>
      <geom name="link1_geom" type="capsule" fromto="0 0 0 {link_lengths[0]} 0 0" size="0.025" rgba="0.2 0.35 0.8 1"/>
      <body name="link2" pos="{link_lengths[0]} 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1" limited="true" range="{joint_min} {joint_max}" damping="0.1"/>
        <geom name="link2_geom" type="capsule" fromto="0 0 0 {link_lengths[1]} 0 0" size="0.025" rgba="0.2 0.6 0.9 1"/>
        <site name="end_effector" pos="{link_lengths[1]} 0 0" size="0.015" rgba="1 0.2 0.2 1"/>
      </body>
    </body>
    <body name="target" mocap="true" pos="{target[0]} {target[1]} {target[2]}">
      <geom name="target_geom" type="sphere" size="0.025" rgba="0 1 0 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_motor" joint="shoulder" ctrllimited="true" ctrlrange="{-control_limit} {control_limit}" gear="{gear}"/>
    <motor name="elbow_motor" joint="elbow" ctrllimited="true" ctrlrange="{-control_limit} {control_limit}" gear="{gear}"/>
  </actuator>
</mujoco>
""".strip()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        mujoco.mj_resetData(self.model, self.data)
        qpos = sample_initial_qpos(self.np_random, self.task_config, self.training)
        target = sample_target_position(self.np_random, self.task_config, self.training)
        self.target_position = apply_moved_target(target, self.failure_config, self.active_failures)
        self.actuator_scale = sample_actuator_scale(self.np_random, self.task_config, self.training)

        self.data.qpos[:2] = qpos
        self.data.qvel[:2] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.mocap_pos[self.target_mocap_id] = self.target_position
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        end_effector = self._end_effector_position()
        self.trajectory = [end_effector.copy()]
        distance = float(np.linalg.norm(end_effector - self.target_position))
        success = distance < float(self.task_config["success_threshold"])
        self.previous_distance = distance
        self._last_reward_terms = {
            "distance": 0.0,
            "progress": 0.0,
            "success_bonus": 0.0,
            "action_penalty": 0.0,
            "timeout_penalty": 0.0,
        }
        return self._observation_with_noise(), self._info(distance, success, False, self._last_reward_terms)

    def step(self, action: np.ndarray):
        clipped_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        std = action_noise_std(self.failure_config, self.active_failures)
        if std > 0.0:
            clipped_action = np.clip(
                clipped_action + self.np_random.normal(0.0, std, size=clipped_action.shape),
                -1.0,
                1.0,
            )
        failure_scale = weak_actuator_scale(self.failure_config, self.active_failures)
        joint_delta_scale = float(self.task_config["control"]["joint_delta_scale"])
        joint_min, joint_max = self.robot_config["joint_limit"]
        previous_qpos = self.data.qpos[:2].copy()
        new_qpos = np.clip(
            previous_qpos + clipped_action * self.actuator_scale * failure_scale * joint_delta_scale,
            float(joint_min),
            float(joint_max),
        )
        self.data.qpos[:2] = new_qpos
        dt = float(self.task_config["simulation"]["timestep"]) * int(self.task_config["simulation"]["frame_skip"])
        self.data.qvel[:2] = (new_qpos - previous_qpos) / dt
        self.data.ctrl[:] = (
            clipped_action
            * self.actuator_scale
            * failure_scale
            * float(self.robot_config["control_limit"])
        )
        mujoco.mj_forward(self.model, self.data)

        self.step_count += 1
        end_effector = self._end_effector_position()
        distance = float(np.linalg.norm(end_effector - self.target_position))
        success = distance < float(self.task_config["success_threshold"])
        timeout = self.step_count >= int(self.task_config["max_episode_steps"])
        terminated = bool(success)
        truncated = bool(timeout and not success)
        reward, reward_terms = compute_reach_reward(
            distance,
            self.previous_distance,
            success,
            clipped_action,
            timeout,
            self.reward_config,
        )
        self.previous_distance = distance
        self._last_reward_terms = reward_terms
        self.trajectory.append(end_effector.copy())
        return (
            self._observation_with_noise(),
            reward,
            terminated,
            truncated,
            self._info(distance, success, timeout, reward_terms),
        )

    def _observation(self) -> np.ndarray:
        end_effector = self._end_effector_position()
        relative = self.target_position - end_effector
        return np.concatenate(
            [
                self.data.qpos[:2],
                self.data.qvel[:2],
                end_effector,
                self.target_position,
                relative,
            ]
        ).astype(np.float32)

    def _observation_with_noise(self) -> np.ndarray:
        observation = self._observation()
        std = observation_noise_std(
            self.task_config,
            self.failure_config,
            self.active_failures,
            self.training,
        )
        if std > 0.0:
            observation = observation + self.np_random.normal(0.0, std, size=observation.shape).astype(np.float32)
        return observation.astype(np.float32)

    def _end_effector_position(self) -> np.ndarray:
        return self.data.site_xpos[self.end_effector_site_id].copy()

    def _info(
        self,
        distance: float,
        success: bool,
        timeout: bool,
        reward_terms: Mapping[str, float],
    ) -> dict[str, Any]:
        end_effector = self._end_effector_position()
        return {
            "distance": float(distance),
            "success": bool(success),
            "timeout": bool(timeout),
            "reward_terms": {key: float(value) for key, value in reward_terms.items()},
            "target_position": self.target_position.astype(np.float64).copy(),
            "end_effector_position": end_effector.astype(np.float64).copy(),
            "actuator_scale": float(self.actuator_scale),
            "active_failures": sorted(self.active_failures),
        }

    def close(self) -> None:
        return None
