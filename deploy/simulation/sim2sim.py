"""Sim-to-sim playback for the Go2 + ARX L5 policy in native MuJoCo.

This script runs a trained mjlab/RSL-RL checkpoint directly in MuJoCo.  It is
intended as a light deployment sanity check: the observation order, action
scales, joint defaults, and command representation mirror
``Mjlab-Velocity-Flat-Go2arm``.

Example:
  uv run python deploy/simulation/sim2sim.py \
    --checkpoint logs/rsl_rl/unitree_Go2arm_flat/RUN/model_1000.pt
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = (
  REPO_ROOT / "src/mjlab/asset_zoo/robots/go2arm/xmls/go2arm.xml"
)

JOINT_NAMES = (
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

DEFAULT_JOINT_POS = np.array(
  [
    -0.1,
    0.8,
    -1.5,
    0.1,
    0.8,
    -1.5,
    -0.1,
    1.0,
    -1.5,
    0.1,
    1.0,
    -1.5,
    0.0,
    1.55,
    0.95,
    0.45,
    0.0,
    0.0,
  ],
  dtype=np.float32,
)

ACTION_SCALE = np.array(
  [0.25] * 12 + [0.5] * 6,
  dtype=np.float32,
)

HISTORY_LENGTH = 10
OBS_FRAME_DIM = 70
OBS_DIM = HISTORY_LENGTH * OBS_FRAME_DIM
ACTION_DIM = 18
TERM_NAMES = (
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
  "actions",
  "velocity_commands",
  "go2_pose_command",
  "projected_gravity",
)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
  """Multiply quaternions in MuJoCo/wxyz convention."""
  w1, x1, y1, z1 = q1
  w2, x2, y2, z2 = q2
  return np.array(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    dtype=np.float32,
  )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
  return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate world vector ``v`` into the frame represented by ``q``."""
  vq = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
  return quat_mul(quat_mul(quat_conjugate(q), vq), q)[1:]


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate vector ``v`` by quaternion ``q`` in wxyz convention."""
  vq = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
  return quat_mul(quat_mul(q, vq), quat_conjugate(q))[1:]


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
  """Convert a wxyz quaternion to a 3x3 rotation matrix."""
  q = q / np.linalg.norm(q)
  w, x, y, z = q
  return np.array(
    [
      [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
      [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
      [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ],
    dtype=np.float64,
  )


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
  """Quaternion from XYZ fixed-axis Euler angles, returned as wxyz."""
  cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
  cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
  cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
  quat = np.array(
    [
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
    ],
    dtype=np.float32,
  )
  if quat[0] < 0.0:
    quat *= -1.0
  return quat / np.linalg.norm(quat)


def build_actor_from_checkpoint(checkpoint_path: Path) -> nn.Module:
  """Build the deterministic actor MLP from an mjlab/RSL-RL checkpoint."""
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  state_dict = checkpoint.get("actor_state_dict")
  if state_dict is None and "model_state_dict" in checkpoint:
    state_dict = {
      k.replace("actor.", "mlp."): v
      for k, v in checkpoint["model_state_dict"].items()
      if k.startswith("actor.")
    }
  if state_dict is None:
    raise ValueError(f"Could not find actor weights in {checkpoint_path}")

  weight_keys = sorted(
    [k for k in state_dict if k.startswith("mlp.") and k.endswith(".weight")],
    key=lambda name: int(name.split(".")[1]),
  )
  layers: list[nn.Module] = []
  remapped: dict[str, torch.Tensor] = {}
  seq_idx = 0
  for i, weight_key in enumerate(weight_keys):
    layer_idx = weight_key.split(".")[1]
    weight = state_dict[f"mlp.{layer_idx}.weight"]
    bias = state_dict[f"mlp.{layer_idx}.bias"]
    layers.append(nn.Linear(weight.shape[1], weight.shape[0]))
    remapped[f"{seq_idx}.weight"] = weight
    remapped[f"{seq_idx}.bias"] = bias
    seq_idx += 1
    if i != len(weight_keys) - 1:
      layers.append(nn.ELU())
      seq_idx += 1

  actor = nn.Sequential(*layers)
  actor.load_state_dict(remapped)
  actor.eval()
  return actor


def load_policy(policy_path: Path) -> nn.Module:
  """Load either a TorchScript actor or a raw mjlab/RSL-RL checkpoint."""
  try:
    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()
    return policy
  except RuntimeError:
    return build_actor_from_checkpoint(policy_path)


def named_joint_indices(
  model: mujoco.MjModel, names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  joint_ids = np.array(
    [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names],
    dtype=np.int32,
  )
  if np.any(joint_ids < 0):
    missing = [name for name, jid in zip(names, joint_ids, strict=True) if jid < 0]
    raise ValueError(f"Missing joints in MuJoCo model: {missing}")

  qpos_ids = model.jnt_qposadr[joint_ids]
  qvel_ids = model.jnt_dofadr[joint_ids]
  return joint_ids, qpos_ids, qvel_ids


def actuator_indices_for_joints(
  model: mujoco.MjModel, joint_ids: np.ndarray
) -> np.ndarray:
  actuator_ids = []
  for joint_id in joint_ids:
    matches = np.where(model.actuator_trnid[:, 0] == joint_id)[0]
    if len(matches) == 0:
      joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
      raise ValueError(f"No position actuator found for joint {joint_name}")
    actuator_ids.append(int(matches[0]))
  return np.asarray(actuator_ids, dtype=np.int32)


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
  if key_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
  else:
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = np.array([0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0])
  mujoco.mj_forward(model, data)


def load_model(
  xml_path: Path,
  add_floor: bool,
  physics_dt: float,
  iterations: int,
  ls_iterations: int,
  disable_nativeccd: bool,
) -> mujoco.MjModel:
  """Load the robot XML and optionally add a MuJoCo floor.

  The asset XML mirrors mjlab's robot asset and does not contain terrain.
  mjlab adds terrain through the environment config, so native MuJoCo sim2sim
  needs to add its own floor here.
  """
  spec = mujoco.MjSpec.from_file(str(xml_path))
  spec.option.timestep = physics_dt
  spec.option.iterations = iterations
  spec.option.ls_iterations = ls_iterations
  if disable_nativeccd:
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)

  if add_floor:
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [20.0, 20.0, 0.05]
    floor.pos = [0.0, 0.0, 0.0]
    floor.rgba = [0.78, 0.78, 0.78, 1.0]
    floor.friction = [0.8, 0.02, 0.01]
    floor.condim = 6
    floor.priority = 1

  return spec.compile()


def base_velocity_body(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
  base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
  velocity = np.zeros(6, dtype=np.float64)
  mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, velocity, 1)
  # MuJoCo returns spatial velocity as [angular, linear].
  return velocity.astype(np.float32)


def make_ee_command(args: argparse.Namespace, base_z: float) -> np.ndarray:
  """Build training-style ee_pose command: x/y in base frame, z as world target."""
  pos_b = np.array(
    [args.ee_x, args.ee_y, args.ee_z - base_z],
    dtype=np.float32,
  )
  pitch = -math.atan2(pos_b[2], math.sqrt(pos_b[0] ** 2 + pos_b[1] ** 2))
  yaw = math.atan2(pos_b[1], pos_b[0])
  quat_b = quat_from_euler_xyz(
    args.ee_roll,
    pitch + args.ee_pitch_offset,
    yaw + args.ee_yaw_offset,
  )
  return np.concatenate([pos_b, quat_b]).astype(np.float32)


def command_world_pose(
  data: mujoco.MjData,
  args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
  """Return target EE pose in world frame, matching UniformPoseCommand metrics."""
  command_b = make_ee_command(args, base_z=float(data.qpos[2]))
  base_pos = data.qpos[:3].astype(np.float32)
  base_quat = data.qpos[3:7].astype(np.float32)
  target_pos = base_pos + quat_rotate(base_quat, command_b[:3])
  target_pos[2] = args.ee_z
  target_quat = quat_mul(base_quat, command_b[3:])
  return target_pos.astype(np.float64), target_quat.astype(np.float64)


def add_scene_sphere(
  scene: mujoco.MjvScene,
  pos: np.ndarray,
  radius: float,
  rgba: tuple[float, float, float, float],
) -> None:
  if scene.ngeom >= scene.maxgeom:
    return
  geom = scene.geoms[scene.ngeom]
  mujoco.mjv_initGeom(
    geom,
    mujoco.mjtGeom.mjGEOM_SPHERE,
    np.array([radius, 0.0, 0.0], dtype=np.float64),
    pos.astype(np.float64),
    np.eye(3, dtype=np.float64).reshape(-1),
    np.array(rgba, dtype=np.float32),
  )
  scene.ngeom += 1


def add_scene_frame(
  scene: mujoco.MjvScene,
  pos: np.ndarray,
  quat: np.ndarray,
  scale: float,
  radius: float,
  alpha: float,
) -> None:
  rot = quat_to_matrix(quat.astype(np.float32))
  colors = (
    (1.0, 0.0, 0.0, alpha),
    (0.0, 0.8, 0.0, alpha),
    (0.1, 0.2, 1.0, alpha),
  )
  for axis, rgba in enumerate(colors):
    if scene.ngeom >= scene.maxgeom:
      return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
      geom,
      mujoco.mjtGeom.mjGEOM_CAPSULE,
      np.zeros(3, dtype=np.float64),
      np.zeros(3, dtype=np.float64),
      np.eye(3, dtype=np.float64).reshape(-1),
      np.array(rgba, dtype=np.float32),
    )
    start = pos.astype(np.float64)
    end = start + rot[:, axis] * scale
    mujoco.mjv_connector(
      geom,
      mujoco.mjtGeom.mjGEOM_CAPSULE,
      radius,
      start,
      end.astype(np.float64),
    )
    geom.rgba[:] = np.array(rgba, dtype=np.float32)
    scene.ngeom += 1


def draw_ee_visualization(
  viewer: mujoco.viewer.Handle,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  site_id: int,
  args: argparse.Namespace,
) -> None:
  """Draw target and current EE markers in the passive MuJoCo viewer."""
  if not args.visualize_ee:
    return

  scene = viewer.user_scn
  scene.ngeom = 0

  target_pos, target_quat = command_world_pose(data, args)
  ee_pos = data.site_xpos[site_id].copy()
  ee_mat = data.site_xmat[site_id].reshape(3, 3).copy()
  ee_quat = np.empty(4, dtype=np.float64)
  mujoco.mju_mat2Quat(ee_quat, ee_mat.reshape(-1))

  add_scene_sphere(scene, target_pos, args.viz_sphere_radius, (0.1, 1.0, 0.1, 0.85))
  add_scene_frame(scene, target_pos, target_quat, args.viz_frame_scale, 0.006, 0.9)
  add_scene_sphere(scene, ee_pos, args.viz_sphere_radius * 0.75, (0.0, 0.9, 1.0, 0.75))
  add_scene_frame(scene, ee_pos, ee_quat, args.viz_frame_scale * 0.75, 0.004, 0.75)


def build_observation_terms(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  qpos_ids: np.ndarray,
  qvel_ids: np.ndarray,
  last_action: np.ndarray,
  velocity_command: np.ndarray,
  args: argparse.Namespace,
) -> np.ndarray:
  base_vel = base_velocity_body(model, data)
  base_ang_vel = base_vel[:3]
  joint_pos_rel = data.qpos[qpos_ids].astype(np.float32) - DEFAULT_JOINT_POS
  joint_vel_rel = data.qvel[qvel_ids].astype(np.float32)
  ee_command = make_ee_command(args, base_z=float(data.qpos[2]))
  projected_gravity = quat_rotate_inverse(
    data.qpos[3:7].astype(np.float32),
    np.array([0.0, 0.0, -1.0], dtype=np.float32),
  )

  return {
    "base_ang_vel": base_ang_vel.astype(np.float32),
    "joint_pos": joint_pos_rel.astype(np.float32),
    "joint_vel": joint_vel_rel.astype(np.float32),
    "actions": last_action.astype(np.float32),
    "velocity_commands": velocity_command.astype(np.float32),
    "go2_pose_command": ee_command.astype(np.float32),
    "projected_gravity": projected_gravity.astype(np.float32),
  }


def build_actor_observation(
  history: dict[str, deque[np.ndarray]],
) -> np.ndarray:
  """Pack observation history exactly like mjlab's ObservationManager.

  mjlab stores history per observation term, flattens each term's 10-frame
  buffer, then concatenates terms in config order.  Packing by full frames
  would still produce 700 numbers, but every feature would be in the wrong
  place for the trained policy.
  """
  return np.concatenate(
    [np.concatenate(list(history[name]), axis=0) for name in TERM_NAMES],
    axis=0,
  ).astype(np.float32)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
  parser.add_argument("--duration", type=float, default=60.0)
  parser.add_argument("--control-dt", type=float, default=0.02)
  parser.add_argument("--physics-dt", type=float, default=0.005)
  parser.add_argument("--iterations", type=int, default=10)
  parser.add_argument("--ls-iterations", type=int, default=20)
  parser.add_argument(
    "--disable-nativeccd",
    action=argparse.BooleanOptionalAction,
    default=True,
  )
  parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--floor", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--action-clip", type=float, default=1.0)
  parser.add_argument("--lin-vel-x", type=float, default=0.3)
  parser.add_argument("--lin-vel-y", type=float, default=0.0)
  parser.add_argument("--ang-vel-z", type=float, default=0.0)
  parser.add_argument("--ee-x", type=float, default=0.48)
  parser.add_argument("--ee-y", type=float, default=0.0)
  parser.add_argument("--ee-z", type=float, default=0.36)
  parser.add_argument("--ee-roll", type=float, default=0.0)
  parser.add_argument("--ee-pitch-offset", type=float, default=0.0)
  parser.add_argument("--ee-yaw-offset", type=float, default=0.0)
  parser.add_argument("--visualize-ee", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--viz-frame-scale", type=float, default=0.16)
  parser.add_argument("--viz-sphere-radius", type=float, default=0.025)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  model = load_model(
    args.xml,
    add_floor=args.floor,
    physics_dt=args.physics_dt,
    iterations=args.iterations,
    ls_iterations=args.ls_iterations,
    disable_nativeccd=args.disable_nativeccd,
  )
  data = mujoco.MjData(model)
  reset_to_home(model, data)

  joint_ids, qpos_ids, qvel_ids = named_joint_indices(model, JOINT_NAMES)
  actuator_ids = actuator_indices_for_joints(model, joint_ids)
  site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
  if site_id < 0:
    raise ValueError("Missing site 'end_effector' in MuJoCo model.")
  policy = load_policy(args.checkpoint)

  velocity_command = np.array(
    [args.lin_vel_x, args.lin_vel_y, args.ang_vel_z],
    dtype=np.float32,
  )
  last_action = np.zeros(ACTION_DIM, dtype=np.float32)
  obs_history: dict[str, deque[np.ndarray]] = {
    name: deque(maxlen=HISTORY_LENGTH) for name in TERM_NAMES
  }
  first_obs = build_observation_terms(
    model, data, qpos_ids, qvel_ids, last_action, velocity_command, args
  )
  for name in TERM_NAMES:
    for _ in range(HISTORY_LENGTH):
      obs_history[name].append(first_obs[name])

  control_steps = max(1, int(round(args.control_dt / model.opt.timestep)))
  total_steps = int(args.duration / model.opt.timestep)

  def policy_step() -> None:
    nonlocal last_action
    obs = build_observation_terms(
      model, data, qpos_ids, qvel_ids, last_action, velocity_command, args
    )
    for name in TERM_NAMES:
      obs_history[name].append(obs[name])
    actor_obs = build_actor_observation(obs_history)
    if actor_obs.shape[0] != OBS_DIM:
      raise RuntimeError(f"Expected obs dim {OBS_DIM}, got {actor_obs.shape[0]}")
    with torch.no_grad():
      action = policy(torch.from_numpy(actor_obs).unsqueeze(0)).squeeze(0).numpy()
    last_action = np.clip(action, -args.action_clip, args.action_clip).astype(np.float32)
    target_joint_pos = DEFAULT_JOINT_POS + last_action * ACTION_SCALE
    data.ctrl[actuator_ids] = target_joint_pos

  if args.render:
    with mujoco.viewer.launch_passive(model, data) as viewer:
      for step in range(total_steps):
        step_start = time.time()
        if step % control_steps == 0:
          policy_step()
        mujoco.mj_step(model, data)
        draw_ee_visualization(viewer, model, data, site_id, args)
        viewer.sync()
        sleep_time = model.opt.timestep - (time.time() - step_start)
        if sleep_time > 0.0:
          time.sleep(sleep_time)
        if not viewer.is_running():
          break
  else:
    for step in range(total_steps):
      if step % control_steps == 0:
        policy_step()
      mujoco.mj_step(model, data)


if __name__ == "__main__":
  main()
