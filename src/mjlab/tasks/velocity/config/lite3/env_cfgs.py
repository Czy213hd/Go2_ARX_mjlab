"""Lite3 velocity environment configurations."""

from collections import OrderedDict

from mjlab.asset_zoo.robots import (
  LITE3_ACTION_SCALE,
  get_lite3_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def _lite3_base_cfg() -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": get_lite3_robot_cfg()}

  cfg.observations["actor"].terms["base_lin_vel"].params["sensor_name"] = (
    "robot/TORSO_site_linvel"
  )
  cfg.observations["actor"].terms["base_ang_vel"].params["sensor_name"] = (
    "robot/TORSO_site_angvel"
  )
  cfg.observations["critic"].terms["base_lin_vel"].params["sensor_name"] = (
    "robot/TORSO_site_linvel"
  )
  cfg.observations["critic"].terms["base_ang_vel"].params["sensor_name"] = (
    "robot/TORSO_site_angvel"
  )
  cfg.observations["actor"].terms.pop("base_lin_vel", None)
  cfg.observations["critic"].terms.pop("base_lin_vel", None)
  cfg.observations["actor"].terms = OrderedDict(
    (
      ("base_ang_vel", cfg.observations["actor"].terms["base_ang_vel"]),
      ("projected_gravity", cfg.observations["actor"].terms["projected_gravity"]),
      ("command", cfg.observations["actor"].terms["command"]),
      ("joint_pos", cfg.observations["actor"].terms["joint_pos"]),
      ("joint_vel", cfg.observations["actor"].terms["joint_vel"]),
      ("actions", cfg.observations["actor"].terms["actions"]),
    )
  )
  cfg.observations["critic"].terms = OrderedDict(
    (
      ("base_ang_vel", cfg.observations["critic"].terms["base_ang_vel"]),
      ("projected_gravity", cfg.observations["critic"].terms["projected_gravity"]),
      ("command", cfg.observations["critic"].terms["command"]),
      ("joint_pos", cfg.observations["critic"].terms["joint_pos"]),
      ("joint_vel", cfg.observations["critic"].terms["joint_vel"]),
      ("actions", cfg.observations["critic"].terms["actions"]),
    )
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = LITE3_ACTION_SCALE

  cfg.viewer.body_name = "TORSO"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -8.0

  # Start conservative: remove terms requiring robot-specific contact and site
  # naming until Lite3 geoms/sites are fully validated.
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "foot_height_scan"
  )
  cfg.observations["critic"].terms.pop("foot_height", None)
  cfg.observations["critic"].terms.pop("foot_air_time", None)
  cfg.observations["critic"].terms.pop("foot_contact", None)
  cfg.observations["critic"].terms.pop("foot_contact_forces", None)

  cfg.rewards.pop("air_time", None)
  cfg.rewards.pop("foot_clearance", None)
  cfg.rewards.pop("foot_swing_height", None)
  cfg.rewards.pop("foot_slip", None)
  cfg.rewards.pop("soft_landing", None)
  cfg.rewards.pop("angular_momentum", None)

  cfg.events.pop("foot_friction", None)
  cfg.events.pop("base_com", None)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*_HipX_joint": 0.05,
    r".*_HipY_joint": 0.05,
    r".*_Knee_joint": 0.08,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_HipX_joint": 0.2,
    r".*_HipY_joint": 0.3,
    r".*_Knee_joint": 0.4,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_HipX_joint": 0.25,
    r".*_HipY_joint": 0.4,
    r".*_Knee_joint": 0.6,
  }

  # Velocity command sampling (training only; play mode overrides below).
  # More +lin_vel_x "forward-only" episodes so the policy tracks W / +vx in sim2sim.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_forward_envs = 0.45
  twist_cmd.rel_heading_envs = 0.15

  return cfg


def lite3_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Lite3 rough terrain velocity configuration."""
  cfg = _lite3_base_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("encoder_bias", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["reset_base"].params["pose_range"] = {
      "x": (0.0, 0.0),
      "y": (0.0, 0.0),
      "z": (0.0, 0.0),
      "yaw": (0.0, 0.0),
    }
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.rel_standing_envs = 0.1
    twist_cmd.rel_heading_envs = 0.0
    twist_cmd.rel_forward_envs = 0.0
    # Play: wider command ranges than training defaults for interactive testing.
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.8, 0.8)

    if (
      cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None
    ):
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def lite3_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Lite3 flat terrain velocity configuration."""
  cfg = _lite3_base_cfg()

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove terrain-scan terms for flat setup.
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("encoder_bias", None)
    cfg.curriculum = {}
    cfg.events["reset_base"].params["pose_range"] = {
      "x": (0.0, 0.0),
      "y": (0.0, 0.0),
      "z": (0.0, 0.0),
      "yaw": (0.0, 0.0),
    }
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.rel_standing_envs = 0.1
    twist_cmd.rel_heading_envs = 0.0
    twist_cmd.rel_forward_envs = 0.0
    # Play: wider command ranges than training defaults for interactive testing.
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.8, 0.8)

  return cfg
