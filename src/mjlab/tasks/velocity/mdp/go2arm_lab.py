"""Go2Arm_Lab-style MDP pieces for the Go2 + arm velocity task."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  combine_frame_transforms,
  compute_pose_error,
  matrix_from_quat,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_mul,
  quat_unique,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


GO2ARM_ORDERED_JOINTS = (
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
  "joint1",
  "joint2",
  "joint3",
  "joint4",
  "joint5",
  "joint6",
)

GO2ARM_LEG_JOINTS = GO2ARM_ORDERED_JOINTS[:12]
GO2ARM_ARM_JOINTS = GO2ARM_ORDERED_JOINTS[12:]


def _ordered_joint_ids(robot: Entity, names: tuple[str, ...] = GO2ARM_ORDERED_JOINTS):
  ids, _ = robot.find_joints(names, preserve_order=True)
  return ids


def go2arm_joint_pos_rel(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = _ordered_joint_ids(robot)
  return robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]


def go2arm_joint_vel_rel(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = _ordered_joint_ids(robot)
  return robot.data.joint_vel[:, ids] - robot.data.default_joint_vel[:, ids]


def go2arm_joint_torques(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = _ordered_joint_ids(robot)
  return robot.data.qfrc_actuator[:, ids]


def go2arm_feet_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  return sensor.compute_first_contact(env.step_dt).float()


def go2arm_mass_base(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  body_ids, _ = robot.find_bodies(("base",), preserve_order=True)
  mass = robot.data.model.body_subtreemass[:, robot.indexing.body_ids[body_ids]]
  return mass[:, :1]


def go2arm_mass_ee(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  site_ids, _ = robot.find_sites(("end_effector",), preserve_order=True)
  body_id = robot.data.model.site_bodyid[robot.indexing.site_ids[site_ids[0]]]
  local_body_ids = (robot.indexing.body_ids == body_id).nonzero().flatten()
  if len(local_body_ids) == 0:
    return torch.zeros(env.num_envs, 1, device=env.device)
  mass = robot.data.model.body_subtreemass[:, body_id]
  return mass.unsqueeze(1)


class UniformPoseCommand(CommandTerm):
  cfg: UniformPoseCommandCfg

  def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    site_ids, _ = self.robot.find_sites((cfg.site_name,), preserve_order=True)
    self.site_id = site_ids[0]
    self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
    self.pose_command_b[:, 3] = 1.0
    self.pose_command_w_z = torch.zeros(self.num_envs, 1, device=self.device)
    self.pose_command_w = torch.zeros_like(self.pose_command_b)
    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.pose_command_b

  def _update_metrics(self) -> None:
    self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
      self.robot.data.root_link_pos_w,
      self.robot.data.root_link_quat_w,
      self.pose_command_b[:, :3],
      self.pose_command_b[:, 3:],
    )
    self.pose_command_w[:, 2] = self.pose_command_w_z[:, 0]
    pos_error, rot_error = compute_pose_error(
      self.pose_command_w[:, :3],
      self.pose_command_w[:, 3:],
      self.robot.data.site_pos_w[:, self.site_id],
      self.robot.data.site_quat_w[:, self.site_id],
    )
    self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
    self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    r = torch.empty(n, device=self.device)
    r1 = torch.empty(1, device=self.device)
    count = torch.tensor(
      self._env.common_step_counter
      / max(self.cfg.curriculum_coeff * self.cfg.num_steps_per_env, 1),
      device=self.device,
    )
    init_w = torch.clamp(1.0 - count, 0.0, 1.0)
    final_w = torch.clamp(count, 0.0, 1.0)

    self.pose_command_b[env_ids, 0] = (
      r.uniform_(*self.cfg.ranges_init.pos_x) * init_w
      + r.uniform_(*self.cfg.ranges_final.pos_x) * final_w
    )
    self.pose_command_b[env_ids, 1] = (
      r.uniform_(*self.cfg.ranges_init.pos_y) * init_w
      + r.uniform_(*self.cfg.ranges_final.pos_y) * final_w
    )
    self.pose_command_w_z[env_ids, 0] = (
      r.uniform_(*self.cfg.ranges_init.pos_z) * init_w
      + r.uniform_(*self.cfg.ranges_final.pos_z) * final_w
    )
    self.pose_command_b[env_ids, 2] = (
      self.pose_command_w_z[env_ids, 0] - self.robot.data.root_link_pos_w[env_ids, 2]
    )

    for i in env_ids.tolist():
      length_arm = torch.norm(self.pose_command_b[i, :3])
      while (
        (length_arm > 0.7)
        or (length_arm < 0.3)
        or (
          self.pose_command_b[i, 0] < 0.45
          and torch.abs(self.pose_command_b[i, 1]) < 0.2
        )
      ):
        self.pose_command_b[i, 0] = (
          r1.uniform_(*self.cfg.ranges_init.pos_x) * init_w
          + r1.uniform_(*self.cfg.ranges_final.pos_x) * final_w
        )
        self.pose_command_b[i, 1] = (
          r1.uniform_(*self.cfg.ranges_init.pos_y) * init_w
          + r1.uniform_(*self.cfg.ranges_final.pos_y) * final_w
        )
        self.pose_command_w_z[i, 0] = (
          r1.uniform_(*self.cfg.ranges_init.pos_z) * init_w
          + r1.uniform_(*self.cfg.ranges_final.pos_z) * final_w
        )
        self.pose_command_b[i, 2] = (
          self.pose_command_w_z[i, 0] - self.robot.data.root_link_pos_w[i, 2]
        )
        length_arm = torch.norm(self.pose_command_b[i, :3])

    euler = torch.zeros(n, 3, device=self.device)
    delta_x = self.pose_command_b[env_ids, 0]
    delta_y = self.pose_command_b[env_ids, 1]
    delta_z = self.pose_command_b[env_ids, 2]
    euler[:, 0] = (
      r.uniform_(*self.cfg.ranges_init.roll) * init_w
      + r.uniform_(*self.cfg.ranges_final.roll) * final_w
    )
    euler[:, 1] = (
      -torch.atan2(delta_z, torch.sqrt(delta_x**2 + delta_y**2))
      + r.uniform_(*self.cfg.ranges.pitch) * init_w
      + r.uniform_(*self.cfg.ranges_final.pitch) * final_w
    )
    euler[:, 2] = (
      torch.atan2(delta_y, delta_x)
      + r.uniform_(*self.cfg.ranges_init.yaw) * init_w
      + r.uniform_(*self.cfg.ranges_final.yaw) * final_w
    )
    quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
    self.pose_command_b[env_ids, 3:] = quat_unique(quat)

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    self._update_metrics()
    command_positions = self.pose_command_w[:, :3].detach().cpu().numpy()
    command_rotation_matrices = matrix_from_quat(
      self.pose_command_w[:, 3:]
    ).detach().cpu().numpy()
    ee_positions = self.robot.data.site_pos_w[:, self.site_id].detach().cpu().numpy()
    ee_rotation_matrices = matrix_from_quat(
      self.robot.data.site_quat_w[:, self.site_id]
    ).detach().cpu().numpy()
    for batch in env_indices:
      visualizer.add_frame(
        position=command_positions[batch],
        rotation_matrix=command_rotation_matrices[batch],
        scale=0.16,
        label=f"ee_pose_command_{batch}",
        axis_radius=0.008,
        alpha=0.9,
      )
      visualizer.add_frame(
        position=ee_positions[batch],
        rotation_matrix=ee_rotation_matrices[batch],
        scale=0.12,
        label=f"ee_pose_current_{batch}",
        axis_radius=0.006,
        alpha=0.8,
        axis_colors=((1.0, 0.35, 0.35), (0.35, 1.0, 0.35), (0.35, 0.55, 1.0)),
      )


@dataclass(kw_only=True)
class UniformPoseCommandCfg(CommandTermCfg):
  entity_name: str
  site_name: str
  curriculum_coeff: float = 1000.0
  num_steps_per_env: int = 24

  @dataclass
  class Ranges:
    pos_x: tuple[float, float]
    pos_y: tuple[float, float]
    pos_z: tuple[float, float]
    roll: tuple[float, float]
    pitch: tuple[float, float]
    yaw: tuple[float, float]

  ranges: Ranges
  ranges_init: Ranges
  ranges_final: Ranges

  def build(self, env: ManagerBasedRlEnv) -> UniformPoseCommand:
    return UniformPoseCommand(self, env)


def position_command_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  des_pos_b = command[:, :3]
  des_pos_w, _ = combine_frame_transforms(
    robot.data.root_link_pos_w, robot.data.root_link_quat_w, des_pos_b
  )
  des_pos_w[:, 2] = des_pos_b[:, 2] + robot.data.root_link_pos_w[:, 2]
  curr_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids[0], :3]
  return torch.exp(-torch.sum(torch.abs(curr_pos_w - des_pos_w) / std, dim=1))


def orientation_command_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  des_quat_w = quat_mul(robot.data.root_link_quat_w, command[:, 3:7])
  curr_quat_w = robot.data.site_quat_w[:, asset_cfg.site_ids[0]]
  return quat_error_magnitude(curr_quat_w, des_quat_w)


def action_rate_l2_arm(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum((env.action_manager.action[:, 12:] - env.action_manager.prev_action[:, 12:]) ** 2, dim=1)


def arm_action_smoothness_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.linalg.norm(env.action_manager.action[:, 12:] - env.action_manager.prev_action[:, 12:], dim=1)


def action_rate_l2_go2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum((env.action_manager.action[:, :12] - env.action_manager.prev_action[:, :12]) ** 2, dim=1)


def leg_action_smoothness_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.linalg.norm(env.action_manager.action[:, :12] - env.action_manager.prev_action[:, :12], dim=1)


def track_lin_vel_xy_exp(env: ManagerBasedRlEnv, std: float, command_name: str) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  error = torch.sum((env.command_manager.get_command(command_name)[:, :2] - robot.data.root_link_lin_vel_b[:, :2]) ** 2, dim=1)
  return torch.exp(-error / std)


def track_ang_vel_z_exp(env: ManagerBasedRlEnv, std: float, command_name: str) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  error = (env.command_manager.get_command(command_name)[:, 2] - robot.data.root_link_ang_vel_b[:, 2]) ** 2
  return torch.exp(-error / std**2)


def lin_vel_z_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["robot"].data.root_link_lin_vel_b[:, 2] ** 2


def ang_vel_xy_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(env.scene["robot"].data.root_link_ang_vel_b[:, :2] ** 2, dim=1)


def joint_torques_l2_go2(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = _ordered_joint_ids(robot, GO2ARM_LEG_JOINTS)
  return torch.sum(robot.data.qfrc_actuator[:, ids] ** 2, dim=1)


def joint_acc_l2_go2(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = _ordered_joint_ids(robot, GO2ARM_LEG_JOINTS)
  return torch.sum(robot.data.joint_acc[:, ids] ** 2, dim=1)


def feet_air_time(env: ManagerBasedRlEnv, command_name: str, sensor_name: str, threshold: float) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  first_contact = sensor.compute_first_contact(env.step_dt)
  assert sensor.data.last_air_time is not None
  reward = torch.sum((sensor.data.last_air_time - threshold) * first_contact, dim=1)
  reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
  return reward


def standing_feet_contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  force_threshold: float,
  command_threshold: float,
  foot_indices: tuple[int, ...] = (2, 3),
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  contact_force = torch.norm(sensor.data.force[:, foot_indices, :], dim=-1)
  command = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
  force = torch.min(contact_force, dim=1).values.clamp(0.0, force_threshold)
  return torch.where(command < command_threshold, 2.0 * force, force)


def joint_deviation_l1(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.abs(robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)


def base_height_l2(env: ManagerBasedRlEnv, target_height: float) -> torch.Tensor:
  height = torch.clamp(env.scene["robot"].data.root_link_pos_w[:, 2], max=0.4)
  return (height - target_height) ** 2


def flat_orientation_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(env.scene["robot"].data.projected_gravity_b[:, :2] ** 2, dim=1)
