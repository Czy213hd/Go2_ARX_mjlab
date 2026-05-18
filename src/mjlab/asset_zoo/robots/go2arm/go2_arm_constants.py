"""Unitree Go2 + ARX L5 arm (go2arm MJCF)."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

GO2ARM_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "go2arm" / "xmls" / "go2arm.xml"
)
assert GO2ARM_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(GO2ARM_XML))


##
# Actuators.
#
# Legs and arm: MJCF ``<position>`` PD actuators, wrapped with
# ``XmlActuatorCfg`` so RL uses joint-position targets for both groups.
##

# Matches Go2Arm_Lab: leg position PD with 40.0 stiffness / 1.0 damping.
GO2ARM_LEG_XML_ACTUATORS = XmlActuatorCfg(
  target_names_expr=(r".*_(hip|thigh|calf)_joint",),
  command_field="position",
)

GO2ARM_ARM_XML_ACTUATORS = XmlActuatorCfg(
  target_names_expr=(r"^joint[1-6]$",),
  command_field="position",
)

##
# Keyframe (matches ``go2arm.xml`` ``<key name="home" ...>``).
#
# ``pos`` / ``rot`` are the floating-base root state (world frame): ``pos`` is
# ``(x, y, z)`` of the free joint; ``rot`` is quaternion ``(w, x, y, z)`` in MuJoCo
# convention, same order as the first 7 numbers of ``qpos`` in that keyframe.
#
# Leg joint defaults follow Go2Arm_Lab: left hips 0.1, right hips -0.1,
# front thighs 0.8, rear thighs 1.0, calves -1.5.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.35),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    ".*L_hip_joint": 0.1,
    ".*R_hip_joint": -0.1,
    "F[LR]_thigh_joint": 0.8,
    "R[LR]_thigh_joint": 1.0,
    ".*_calf_joint": -1.5,
    "joint1": 0.0,
    "joint2": 1.55,
    "joint3": 0.95,
    "joint4": 0.45,
    "joint5": 0.0,
    "joint6": 0.0,
    "joint7": 0.044,
    "joint8": -0.044,
  },
  joint_vel={".*": 0.0},
)

##
# Collision.
##

_foot_geom_re = r"^(FL|FR|RL|RR)$"

# Feet only: named foot spheres (see ``go2arm.xml`` / ``go2.xml``).
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_geom_re,),
  contype=0,
  conaffinity=1,
  condim=6,
  priority=1,
  friction=(0.8, 0.02, 0.01),
  solimp=(0.015, 1.0, 0.022),
)

# All geoms (including unnamed collision meshes); feet get condim 6 + friction.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  condim={_foot_geom_re: 6, ".*": 1},
  priority={_foot_geom_re: 1},
  friction={_foot_geom_re: (0.8, 0.02, 0.01)},
  solref=(0.01, 1),
  disable_other_geoms=False,
)

##
# Articulation and robot cfg.
##

GO2ARM_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2ARM_LEG_XML_ACTUATORS,
    GO2ARM_ARM_XML_ACTUATORS,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2arm_robot_cfg() -> EntityCfg:
  """Return a fresh Go2 + ARX L5 ``EntityCfg`` (safe to share across configs)."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2ARM_ARTICULATION,
  )


# Maps normalized policy actions in ``[-1, 1]`` to position residuals (rad).
GO2ARM_LEG_ACTION_SCALE: dict[str, float] = {
  r".*_hip_joint": 0.25,
  r".*_thigh_joint": 0.25,
  r".*_calf_joint": 0.25,
}

GO2ARM_ARM_ACTION_SCALE: dict[str, float] = {
  "joint1": 0.5,
  "joint2": 0.7,
  "joint3": 0.7,
  "joint4": 0.7,
  "joint5": 0.5,
  "joint6": 0.5,
}

# Backward-compatible aliases for existing imports.
GO2ARM_LEG_EFFORT_SCALE = GO2ARM_LEG_ACTION_SCALE
GO2ARM_ACTION_SCALE = {**GO2ARM_LEG_ACTION_SCALE, **GO2ARM_ARM_ACTION_SCALE}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_go2arm_robot_cfg())
  viewer.launch(robot.spec.compile())
