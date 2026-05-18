"""RL configuration for Go2Arm_Lab-style velocity + EE pose task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def _go2arm_ppo_runner_cfg(
  *,
  experiment_name: str,
  max_iterations: int,
  save_interval: int,
) -> RslRlOnPolicyRunnerCfg:
  """Create a mjlab-compatible PPO config using Go2Arm_Lab hyperparameters."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=save_interval,
    num_steps_per_env=24,
    max_iterations=max_iterations,
  )


def go2arm_rough_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _go2arm_ppo_runner_cfg(
    experiment_name="unitree_Go2arm_rough",
    max_iterations=10_000,
    save_interval=500,
  )


def go2arm_flat_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _go2arm_ppo_runner_cfg(
    experiment_name="unitree_Go2arm_flat",
    max_iterations=15_000,
    save_interval=100,
  )


def go2arm_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return go2arm_rough_ppo_runner_cfg()
