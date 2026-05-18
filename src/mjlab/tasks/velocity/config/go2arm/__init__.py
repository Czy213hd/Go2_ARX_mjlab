from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  go2arm_flat_env_cfg,
  go2arm_rough_env_cfg,
)
from .rl_cfg import go2arm_flat_ppo_runner_cfg, go2arm_rough_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Go2arm",
  env_cfg=go2arm_rough_env_cfg(),
  play_env_cfg=go2arm_rough_env_cfg(play=True),
  rl_cfg=go2arm_rough_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Go2arm",
  env_cfg=go2arm_flat_env_cfg(),
  play_env_cfg=go2arm_flat_env_cfg(play=True),
  rl_cfg=go2arm_flat_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
