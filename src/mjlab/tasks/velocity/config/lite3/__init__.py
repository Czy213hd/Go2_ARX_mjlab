from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  lite3_flat_env_cfg,
  lite3_rough_env_cfg,
)
from .rl_cfg import lite3_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Lite3",
  env_cfg=lite3_rough_env_cfg(),
  play_env_cfg=lite3_rough_env_cfg(play=True),
  rl_cfg=lite3_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Lite3",
  env_cfg=lite3_flat_env_cfg(),
  play_env_cfg=lite3_flat_env_cfg(play=True),
  rl_cfg=lite3_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
