"""Keyboard-controlled sim-to-sim playback for Go2 + ARX L5 in MuJoCo.

This is an interactive version of ``sim2sim.py``.  It loads the same policy and
observation/action pipeline, but lets you change base velocity and EE pose
commands from the keyboard while the MuJoCo viewer is running.

Example:
  uv run python deploy/simulation/sim2sim_keyboard.py \
    --checkpoint logs/rsl_rl/unitree_Go2arm_flat/RUN/model_6000.pt
"""

from __future__ import annotations

import argparse
import sys
import termios
import threading
import time
import tty
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import mujoco
import mujoco.viewer
import numpy as np
import torch

from sim2sim import (
  ACTION_DIM,
  ACTION_SCALE,
  DEFAULT_JOINT_POS,
  DEFAULT_XML,
  HISTORY_LENGTH,
  JOINT_NAMES,
  OBS_DIM,
  TERM_NAMES,
  actuator_indices_for_joints,
  build_actor_observation,
  build_observation_terms,
  command_world_pose,
  draw_ee_visualization,
  load_model,
  load_policy,
  named_joint_indices,
  reset_to_home,
)


COMMAND_LIMITS = {
  "lin_vel_x": (-0.3, 0.8),
  "lin_vel_y": (-0.5, 0.5),
  "ang_vel_z": (-0.8, 0.8),
  "ee_x": (0.40, 0.60),
  "ee_y": (-0.30, 0.30),
  "ee_z": (0.22, 0.55),
  "ee_roll": (-0.35, 0.35),
  "ee_pitch_offset": (-0.35, 0.35),
  "ee_yaw_offset": (-0.35, 0.35),
}


HELP = """
Keyboard controls
-----------------
Base velocity:
  W / S : lin_vel_x +/-
  A / D : lin_vel_y +/-
  Q / E : yaw rate +/-
  Space : zero base velocity

End-effector position:
  I / K : ee_x +/-
  J / L : ee_y +/-
  U / O : ee_z +/-

End-effector orientation offsets:
  R / F : pitch offset +/-
  T / G : yaw offset +/-
  Z / X : roll +/-

Other:
  C : reset EE command
  V : print current command
  H : print this help
  Esc / Ctrl-C : quit
"""


def clamp(value: float, key: str) -> float:
  lo, hi = COMMAND_LIMITS[key]
  return min(max(value, lo), hi)


def print_command(command: SimpleNamespace) -> None:
  target_pos, target_quat = command_world_pose(
    command.data,
    command,
  )
  print(
    "command | "
    f"vel=({command.lin_vel_x:+.2f}, {command.lin_vel_y:+.2f}, {command.ang_vel_z:+.2f}) "
    f"ee=({command.ee_x:.3f}, {command.ee_y:+.3f}, {command.ee_z:.3f}) "
    f"rpy_offset=({command.ee_roll:+.2f}, {command.ee_pitch_offset:+.2f}, "
    f"{command.ee_yaw_offset:+.2f}) "
    f"target_w=({target_pos[0]:.3f}, {target_pos[1]:+.3f}, {target_pos[2]:.3f}) "
    f"quat_w=({target_quat[0]:+.3f}, {target_quat[1]:+.3f}, "
    f"{target_quat[2]:+.3f}, {target_quat[3]:+.3f})"
  )


def reset_ee_command(command: SimpleNamespace) -> None:
  command.ee_x = 0.48
  command.ee_y = 0.0
  command.ee_z = 0.36
  command.ee_roll = 0.0
  command.ee_pitch_offset = 0.0
  command.ee_yaw_offset = 0.0


def handle_key(key: str, command: SimpleNamespace, lock: threading.Lock) -> bool:
  """Apply one terminal key command. Returns False when the program should quit."""
  key = key.lower()
  changed = True
  should_continue = True
  with lock:
    if key == "w":
      command.lin_vel_x = clamp(command.lin_vel_x + command.vel_step, "lin_vel_x")
    elif key == "s":
      command.lin_vel_x = clamp(command.lin_vel_x - command.vel_step, "lin_vel_x")
    elif key == "a":
      command.lin_vel_y = clamp(command.lin_vel_y + command.vel_step, "lin_vel_y")
    elif key == "d":
      command.lin_vel_y = clamp(command.lin_vel_y - command.vel_step, "lin_vel_y")
    elif key == "q":
      command.ang_vel_z = clamp(command.ang_vel_z + command.yaw_step, "ang_vel_z")
    elif key == "e":
      command.ang_vel_z = clamp(command.ang_vel_z - command.yaw_step, "ang_vel_z")
    elif key == " ":
      command.lin_vel_x = 0.0
      command.lin_vel_y = 0.0
      command.ang_vel_z = 0.0
    elif key == "i":
      command.ee_x = clamp(command.ee_x + command.ee_pos_step, "ee_x")
    elif key == "k":
      command.ee_x = clamp(command.ee_x - command.ee_pos_step, "ee_x")
    elif key == "j":
      command.ee_y = clamp(command.ee_y + command.ee_pos_step, "ee_y")
    elif key == "l":
      command.ee_y = clamp(command.ee_y - command.ee_pos_step, "ee_y")
    elif key == "u":
      command.ee_z = clamp(command.ee_z + command.ee_pos_step, "ee_z")
    elif key == "o":
      command.ee_z = clamp(command.ee_z - command.ee_pos_step, "ee_z")
    elif key == "r":
      command.ee_pitch_offset = clamp(
        command.ee_pitch_offset + command.ee_rot_step, "ee_pitch_offset"
      )
    elif key == "f":
      command.ee_pitch_offset = clamp(
        command.ee_pitch_offset - command.ee_rot_step, "ee_pitch_offset"
      )
    elif key == "t":
      command.ee_yaw_offset = clamp(
        command.ee_yaw_offset + command.ee_rot_step, "ee_yaw_offset"
      )
    elif key == "g":
      command.ee_yaw_offset = clamp(
        command.ee_yaw_offset - command.ee_rot_step, "ee_yaw_offset"
      )
    elif key == "z":
      command.ee_roll = clamp(command.ee_roll + command.ee_rot_step, "ee_roll")
    elif key == "x":
      command.ee_roll = clamp(command.ee_roll - command.ee_rot_step, "ee_roll")
    elif key == "c":
      reset_ee_command(command)
    elif key == "v":
      changed = False
      print_command(command)
    elif key == "h":
      changed = False
      print(HELP)
    elif key in ("\x03", "\x04", "\x1b"):
      changed = False
      should_continue = False
    else:
      changed = False

    if changed:
      print_command(command)
  return should_continue


@contextmanager
def terminal_raw_mode(enabled: bool):
  if not enabled or not sys.stdin.isatty():
    yield
    return
  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  try:
    tty.setcbreak(fd)
    yield
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def start_terminal_input_thread(
  command: SimpleNamespace,
  lock: threading.Lock,
  stop_event: threading.Event,
) -> threading.Thread | None:
  if not sys.stdin.isatty():
    print("Terminal input disabled: stdin is not a TTY.")
    return None

  def read_keys() -> None:
    while not stop_event.is_set():
      key = sys.stdin.read(1)
      if not key:
        continue
      if not handle_key(key, command, lock):
        stop_event.set()
        break

  thread = threading.Thread(target=read_keys, daemon=True)
  thread.start()
  return thread


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
  parser.add_argument("--duration", type=float, default=1.0e9)
  parser.add_argument("--control-dt", type=float, default=0.02)
  parser.add_argument("--physics-dt", type=float, default=0.005)
  parser.add_argument("--iterations", type=int, default=10)
  parser.add_argument("--ls-iterations", type=int, default=20)
  parser.add_argument(
    "--disable-nativeccd",
    action=argparse.BooleanOptionalAction,
    default=True,
  )
  parser.add_argument("--floor", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--action-clip", type=float, default=1.0)
  parser.add_argument("--lin-vel-x", type=float, default=0.2)
  parser.add_argument("--lin-vel-y", type=float, default=0.0)
  parser.add_argument("--ang-vel-z", type=float, default=0.0)
  parser.add_argument("--ee-x", type=float, default=0.48)
  parser.add_argument("--ee-y", type=float, default=0.0)
  parser.add_argument("--ee-z", type=float, default=0.36)
  parser.add_argument("--ee-roll", type=float, default=0.0)
  parser.add_argument("--ee-pitch-offset", type=float, default=0.0)
  parser.add_argument("--ee-yaw-offset", type=float, default=0.0)
  parser.add_argument("--vel-step", type=float, default=0.05)
  parser.add_argument("--yaw-step", type=float, default=0.05)
  parser.add_argument("--ee-pos-step", type=float, default=0.02)
  parser.add_argument("--ee-rot-step", type=float, default=0.05)
  parser.add_argument("--visualize-ee", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--viz-frame-scale", type=float, default=0.16)
  parser.add_argument("--viz-sphere-radius", type=float, default=0.025)
  parser.add_argument(
    "--terminal-control",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Read command keys from the terminal instead of the MuJoCo viewer.",
  )
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

  command = SimpleNamespace(**vars(args))
  command.data = data
  for key in COMMAND_LIMITS:
    setattr(command, key, clamp(getattr(command, key), key))

  lock = threading.Lock()
  stop_event = threading.Event()
  last_action = np.zeros(ACTION_DIM, dtype=np.float32)
  obs_history = {name: deque(maxlen=HISTORY_LENGTH) for name in TERM_NAMES}

  with lock:
    velocity_command = np.array(
      [command.lin_vel_x, command.lin_vel_y, command.ang_vel_z],
      dtype=np.float32,
    )
    first_obs = build_observation_terms(
      model, data, qpos_ids, qvel_ids, last_action, velocity_command, command
    )
  for name in TERM_NAMES:
    for _ in range(HISTORY_LENGTH):
      obs_history[name].append(first_obs[name])

  control_steps = max(1, int(round(args.control_dt / model.opt.timestep)))
  total_steps = int(args.duration / model.opt.timestep)

  def policy_step() -> None:
    nonlocal last_action
    with lock:
      velocity_command = np.array(
        [command.lin_vel_x, command.lin_vel_y, command.ang_vel_z],
        dtype=np.float32,
      )
      obs = build_observation_terms(
        model, data, qpos_ids, qvel_ids, last_action, velocity_command, command
      )
    for name in TERM_NAMES:
      obs_history[name].append(obs[name])
    actor_obs = build_actor_observation(obs_history)
    if actor_obs.shape[0] != OBS_DIM:
      raise RuntimeError(f"Expected obs dim {OBS_DIM}, got {actor_obs.shape[0]}")
    with torch.no_grad():
      action = policy(torch.from_numpy(actor_obs).unsqueeze(0)).squeeze(0).numpy()
    last_action = np.clip(action, -args.action_clip, args.action_clip).astype(np.float32)
    data.ctrl[actuator_ids] = DEFAULT_JOINT_POS + last_action * ACTION_SCALE

  print(HELP)
  if args.terminal_control:
    print("Focus this terminal for command keys. The MuJoCo window is display-only.")
  else:
    print("Terminal control disabled. Commands stay fixed.")
  print_command(command)

  with terminal_raw_mode(args.terminal_control):
    input_thread = (
      start_terminal_input_thread(command, lock, stop_event)
      if args.terminal_control
      else None
    )
    if args.render:
      with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(total_steps):
          if stop_event.is_set():
            break
          step_start = time.time()
          if step % control_steps == 0:
            policy_step()
          mujoco.mj_step(model, data)
          with lock:
            draw_ee_visualization(viewer, model, data, site_id, command)
          viewer.sync()
          sleep_time = model.opt.timestep - (time.time() - step_start)
          if sleep_time > 0.0:
            time.sleep(sleep_time)
          if not viewer.is_running():
            break
    else:
      for step in range(total_steps):
        if stop_event.is_set():
          break
        if step % control_steps == 0:
          policy_step()
        mujoco.mj_step(model, data)
    stop_event.set()
    if input_thread is not None:
      input_thread.join(timeout=0.2)


if __name__ == "__main__":
  main()
