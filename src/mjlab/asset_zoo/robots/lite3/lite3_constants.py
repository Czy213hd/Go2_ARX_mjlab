"""Lite3 constants and robot entity configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

LITE3_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "lite3" / "xmls" / "Lite3.xml"
)
assert LITE3_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(LITE3_XML))


##
# Actuator config.
##

LITE3_HIPX_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_HipX_joint",),
  stiffness=30.0,
  damping=1.0,
  effort_limit=40.0,
  armature=0.01,
)
LITE3_HIPY_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_HipY_joint",),
  stiffness=30.0,
  damping=1.0,
  effort_limit=40.0,
  armature=0.01,
)
LITE3_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Knee_joint",),
  stiffness=30.0,
  damping=1.0,
  effort_limit=65.0,
  armature=0.01,
)

##
# Initial state.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.34),
  joint_pos={
    r".*_HipX_joint": 0.0,
    r".*_HipY_joint": -0.8,
    r".*_Knee_joint": 1.6,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^[FH][LR]_FOOT_collision$"

# Keep only foot-ground contacts enabled for stable standing and to avoid
# excessive self-collision jitter during early bring-up.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(1.0,),
)

##
# Final config.
##

LITE3_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    LITE3_HIPX_ACTUATOR_CFG,
    LITE3_HIPY_ACTUATOR_CFG,
    LITE3_KNEE_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_lite3_robot_cfg() -> EntityCfg:
  """Get a fresh Lite3 robot configuration."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=LITE3_ARTICULATION,
  )


# Keep action scaling aligned with Lite3 deploy runtime:
# [HipX, HipY, Knee] = [0.125, 0.25, 0.25]
LITE3_ACTION_SCALE: dict[str, float] = {
  r".*_HipX_joint": 0.125,
  r".*_HipY_joint": 0.25,
  r".*_Knee_joint": 0.25,
}
