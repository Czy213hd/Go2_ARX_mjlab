"""Go2 + ARX L5 arm velocity environment configurations."""

import math
from collections import OrderedDict
from dataclasses import replace

from mjlab.asset_zoo.robots import (
  get_go2arm_robot_cfg,
)
from mjlab.asset_zoo.robots.go2arm.go2_arm_constants import (
  GO2ARM_ARM_ACTION_SCALE,
  GO2ARM_LEG_ACTION_SCALE,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformPoseCommandCfg, UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def _go2arm_base_cfg() -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()
  # Go2 MJCF uses geom margins; MuJoCo-Warp rejects margin + native CCD pairs.
  cfg.sim.mujoco = replace(
    cfg.sim.mujoco,
    disableflags=tuple(dict.fromkeys((*cfg.sim.mujoco.disableflags, "nativeccd"))),
  )
  cfg.scene.entities = {"robot": get_go2arm_robot_cfg()}

  cfg.commands = {
    "ee_pose": UniformPoseCommandCfg(
      entity_name="robot",
      site_name="end_effector",
      resampling_time_range=(6.0, 8.0),
      debug_vis=True,
      curriculum_coeff=1000,
      num_steps_per_env=24,
      ranges=UniformPoseCommandCfg.Ranges(
        pos_x=(0.4, 0.6),
        pos_y=(-0.35, 0.35),
        pos_z=(0.1, 0.55),
        roll=(-0.0, 0.0),
        pitch=(-math.pi / 9, math.pi / 9),
        yaw=(-math.pi / 9, math.pi / 9),
      ),
      ranges_final=UniformPoseCommandCfg.Ranges(
        pos_x=(0.4, 0.6),
        pos_y=(-0.35, 0.35),
        pos_z=(0.1, 0.55),
        roll=(-0.0, 0.0),
        pitch=(-math.pi / 9, math.pi / 9),
        yaw=(-math.pi / 9, math.pi / 9),
      ),
      ranges_init=UniformPoseCommandCfg.Ranges(
        pos_x=(0.45, 0.5),
        pos_y=(-0.05, 0.05),
        pos_z=(0.35, 0.4),
        roll=(-0.0, 0.0),
        pitch=(-0.0, 0.0),
        yaw=(-0.0, 0.0),
      ),
    ),
    "base_velocity": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(10.0, 10.0),
      rel_standing_envs=0.1,
      debug_vis=True,
      curriculum_coeff=1000,
      num_steps_per_env=24,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.2, 1.0),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-0.5, 0.5),
      ),
      ranges_final=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.1, 0.8),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-0.5, 0.5),
      ),
      ranges_init=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.1, 0.35),
        lin_vel_y=(-0.1, 0.1),
        ang_vel_z=(-0.1, 0.1),
      ),
    ),
  }

  policy_terms = OrderedDict(
    (
      (
        "base_ang_vel",
        ObservationTermCfg(
          func=mdp.base_ang_vel,
          history_length=10,
        ),
      ),
      (
        "joint_pos",
        ObservationTermCfg(
          func=mdp.go2arm_joint_pos_rel,
          history_length=10,
          noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
      ),
      (
        "joint_vel",
        ObservationTermCfg(
          func=mdp.go2arm_joint_vel_rel,
          history_length=10,
          noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
      ),
      ("actions", ObservationTermCfg(func=mdp.last_action, history_length=10)),
      (
        "velocity_commands",
        ObservationTermCfg(
          func=mdp.generated_commands,
          history_length=10,
          params={"command_name": "base_velocity"},
        ),
      ),
      (
        "go2_pose_command",
        ObservationTermCfg(
          func=mdp.generated_commands,
          history_length=10,
          params={"command_name": "ee_pose"},
        ),
      ),
      (
        "projected_gravity",
        ObservationTermCfg(
          func=mdp.projected_gravity,
          history_length=10,
          noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
      ),
    )
  )
  critic_terms = OrderedDict(policy_terms)
  critic_terms.update(
    (
      ("priv_mass_base", ObservationTermCfg(func=mdp.go2arm_mass_base)),
      ("priv_mass_ee", ObservationTermCfg(func=mdp.go2arm_mass_ee)),
      ("priv_joint_torques", ObservationTermCfg(func=mdp.go2arm_joint_torques)),
      ("priv_base_lin_vel", ObservationTermCfg(func=mdp.base_lin_vel)),
      (
        "priv_feet_contact",
        ObservationTermCfg(
          func=mdp.go2arm_feet_contact,
          params={"sensor_name": "contact_forces"},
        ),
      ),
    )
  )
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base"

  # Go2 MJCF only defines the IMU site (no per-foot sites); drop foot height scan.
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "foot_height_scan"
  )
  cfg.observations["critic"].terms.pop("foot_height", None)
  cfg.observations["critic"].terms.pop("foot_air_time", None)
  cfg.observations["critic"].terms.pop("foot_contact", None)
  cfg.observations["critic"].terms.pop("foot_contact_forces", None)

  foot_geom_names = ("FL", "FR", "RL", "RR")
  contact_forces_cfg = ContactSensorCfg(
    name="contact_forces",
    primary=ContactMatch(
      mode="body",
      pattern=r".*_calf",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
    history_length=3,
  )
  base_arm_ground_contact_cfg = ContactSensorCfg(
    name="base_arm_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=(r"^base$", r"^base_link$", r"^link[1-8]$"),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="maxforce",
    num_slots=1,
  )
  leg_body_ground_contact_cfg = ContactSensorCfg(
    name="leg_body_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=(r".*_hip$", r".*_thigh$"),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="maxforce",
    num_slots=1,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    contact_forces_cfg,
    base_arm_ground_contact_cfg,
    leg_body_ground_contact_cfg,
    self_collision_cfg,
  )

  # Go2Arm_Lab-style position control for both legs and arm.
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(
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
      ),
      scale=GO2ARM_LEG_ACTION_SCALE,
      use_default_offset=True,
      preserve_order=True,
    ),
    "arm_pose": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
      scale=GO2ARM_ARM_ACTION_SCALE,
      use_default_offset=True,
      preserve_order=True,
    ),
  }

  cfg.viewer.body_name = "base"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)

  cfg.rewards = OrderedDict(
    (
      (
        "end_effector_position_tracking",
        RewardTermCfg(
          func=mdp.position_command_error_exp,
          weight=0.8,
          params={
            "asset_cfg": SceneEntityCfg("robot", site_names="end_effector"),
            "command_name": "ee_pose",
            "std": 0.2,
          },
        ),
      ),
      (
        "end_effector_orientation_tracking",
        RewardTermCfg(
          func=mdp.orientation_command_error,
          weight=-0.2,
          params={
            "asset_cfg": SceneEntityCfg("robot", site_names="end_effector"),
            "command_name": "ee_pose",
          },
        ),
      ),
      ("end_effector_action_rate", RewardTermCfg(func=mdp.action_rate_l2_arm, weight=-0.005)),
      ("end_effector_action_smoothness", RewardTermCfg(func=mdp.arm_action_smoothness_penalty, weight=-0.02)),
      (
        "tracking_lin_vel_x_l1",
        RewardTermCfg(
          func=mdp.track_lin_vel_xy_exp,
          weight=3.0,
          params={"command_name": "base_velocity", "std": 0.2},
        ),
      ),
      (
        "track_ang_vel_z_exp",
        RewardTermCfg(
          func=mdp.track_ang_vel_z_exp,
          weight=2.0,
          params={"command_name": "base_velocity", "std": math.sqrt(0.2)},
        ),
      ),
      ("lin_vel_z_l2", RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.5)),
      ("ang_vel_xy_l2", RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.02)),
      ("dof_torques_l2", RewardTermCfg(func=mdp.joint_torques_l2_go2, weight=-2.0e-5)),
      ("dof_acc_l2", RewardTermCfg(func=mdp.joint_acc_l2_go2, weight=-2.5e-7)),
      ("action_rate_l2", RewardTermCfg(func=mdp.action_rate_l2_go2, weight=-0.004)),
      (
        "feet_air_time",
        RewardTermCfg(
          func=mdp.feet_air_time,
          weight=0.5,
          params={
            "sensor_name": "contact_forces",
            "command_name": "base_velocity",
            "threshold": 0.5,
          },
        ),
      ),
      (
        "F_feet_air_time",
        RewardTermCfg(
          func=mdp.feet_air_time,
          weight=0.5,
          params={
            "sensor_name": "contact_forces",
            "command_name": "base_velocity",
            "threshold": 0.5,
          },
        ),
      ),
      (
        "R_feet_air_time",
        RewardTermCfg(
          func=mdp.feet_air_time,
          weight=2.0,
          params={
            "sensor_name": "contact_forces",
            "command_name": "base_velocity",
            "threshold": 0.5,
          },
        ),
      ),
      (
        "foot_contact",
        RewardTermCfg(
          func=mdp.standing_feet_contact_force,
          weight=0.003,
          params={
            "sensor_name": "contact_forces",
            "command_name": "base_velocity",
            "force_threshold": 7.5,
            "command_threshold": 0.1,
          },
        ),
      ),
      (
        "hip_deviation",
        RewardTermCfg(
          func=mdp.joint_deviation_l1,
          weight=-0.1,
          params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_joint",))},
        ),
      ),
      (
        "joint_deviation",
        RewardTermCfg(
          func=mdp.joint_deviation_l1,
          weight=-0.04,
          params={
            "asset_cfg": SceneEntityCfg(
              "robot", joint_names=(".*_thigh_joint", ".*_calf_joint")
            )
          },
        ),
      ),
      ("action_smoothness", RewardTermCfg(func=mdp.leg_action_smoothness_penalty, weight=-0.008)),
      (
        "height_reward",
        RewardTermCfg(func=mdp.base_height_l2, weight=-2.0, params={"target_height": 0.3}),
      ),
      ("flat_orientation_l2", RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0)),
    )
  )

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "base_arm_ground_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": "base_arm_ground_contact"},
    ),
    "leg_body_ground_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": "leg_body_ground_contact"},
    ),
  }
  cfg.curriculum = {
    "flat_ori_modify": CurriculumTermCfg(
      func=envs_mdp.reward_curriculum,
      params={
        "reward_name": "flat_orientation_l2",
        "stages": [{"step": 2000, "weight": -0.0}],
      },
    ),
    "flat_height_modify": CurriculumTermCfg(
      func=envs_mdp.reward_curriculum,
      params={
        "reward_name": "height_reward",
        "stages": [{"step": 4000, "weight": -1.0}],
      },
    ),
  }

  return cfg


def go2arm_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Go2arm rough terrain velocity configuration."""
  cfg = _go2arm_base_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  del cfg.events["foot_friction"]
  cfg.events["foot_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=("FL", "FR", "RL", "RR")),
      "operation": "abs",
      "axes": [0],
      "ranges": (0.3, 1.5),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_spin"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=("FL", "FR", "RL", "RR")),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [1],
      "ranges": (1e-4, 2e-2),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_roll"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=("FL", "FR", "RL", "RR")),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],
      "ranges": (1e-5, 5e-3),
      "shared_random": True,
    },
  )

  cfg.terminations.pop("fell_over", None)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if (
      cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None
    ):
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def go2arm_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Go2arm flat terrain velocity configuration."""
  cfg = go2arm_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  remove_sensors = {"terrain_scan", "self_collision"}
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)
  if "upright" in cfg.rewards:
    cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
  cfg.rewards.pop("self_collisions", None)

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  cfg.curriculum.pop("terrain_levels", None)

  if play:
    base_velocity_cmd = cfg.commands["base_velocity"]
    assert isinstance(base_velocity_cmd, UniformVelocityCommandCfg)
    base_velocity_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    base_velocity_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
