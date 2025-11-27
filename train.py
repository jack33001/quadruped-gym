"""
Training script for quadruped locomotion with Isaac Lab and RSL-RL.

Run with:
    cd /home/jack/Documents/CP Legged Robots/quadruped-gym
    python train.py
"""
import os
import sys

# CRITICAL: Fix for libtorch executable stack issue on newer Linux kernels (6.17+)
# This must be set BEFORE importing torch or any Isaac modules
os.environ["PYTORCH_NVFUSER_DISABLE_FALLBACK"] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

# Isaac Lab / Isaac Sim setup - must come BEFORE any other imports
from isaaclab.app import AppLauncher

# Create app launcher and launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# Now we can import everything else
import pickle
import shutil

import torch
from isaaclab.envs import ManagerBasedRLEnv

from rsl_rl.runners import OnPolicyRunner

from configs import QuadrupedEnvCfg, TrainCfg
from quadruped_env import IsaacLabVecEnvWrapper


class OnPolicyRunnerWithFlush(OnPolicyRunner):
    """
    OnPolicyRunner with explicit TensorBoard flush after each log.
    """

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Override log method to add explicit flush."""
        super().log(locs, width, pad)
        # Force flush to ensure TensorBoard updates live
        if self.writer is not None:
            self.writer.flush()

    def save(self, path: str, infos: dict | None = None) -> None:
        """Override save to also flush writer."""
        super().save(path, infos)
        if self.writer is not None:
            self.writer.flush()


def get_rsl_rl_cfg(train_cfg: TrainCfg, num_envs: int):
    """Convert TrainCfg to RSL-RL config dictionary."""
    return {
        "algorithm": train_cfg.algorithm,
        "init_member_classes": {},
        "policy": train_cfg.policy,
        "runner": {
            "checkpoint": -1,
            "experiment_name": train_cfg.experiment_name,
            "load_run": -1,
            "log_interval": train_cfg.log_interval,
            "max_iterations": train_cfg.max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": train_cfg.run_name,
            "save_interval": train_cfg.save_interval,
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": train_cfg.num_steps_per_env,
        "save_interval": train_cfg.save_interval,
        "empirical_normalization": train_cfg.empirical_normalization,
        "obs_groups": train_cfg.obs_groups,
        "seed": train_cfg.seed,
    }


def main():
    """Main training function."""

    # Load configurations
    env_cfg = QuadrupedEnvCfg()
    train_cfg = TrainCfg()

    # Create Isaac Lab environment
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Wrap for RSL-RL compatibility
    env = IsaacLabVecEnvWrapper(isaac_env)

    # Setup logging
    log_dir = f"logs/{train_cfg.experiment_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Save configurations
    pickle.dump({
        "env_cfg": env_cfg,
        "train_cfg": train_cfg,
    }, open(f"{log_dir}/cfgs.pkl", "wb"))

    # Get RSL-RL config
    train_cfg_dict = get_rsl_rl_cfg(train_cfg, env_cfg.scene.num_envs)

    # Print training info
    print("\n" + "=" * 80)
    print("QUADRUPED RL TRAINING (Isaac Lab)")
    print("=" * 80)
    print(f"Experiment: {train_cfg.experiment_name}")
    print(f"Max iterations: {train_cfg.max_iterations}")
    print(f"Num environments: {env_cfg.scene.num_envs}")
    print(f"Steps per env: {train_cfg.num_steps_per_env}")
    print(f"Log directory: {log_dir}")
    print(f"Device: {env.device}")
    print("=" * 80)
    print("\nTensorBoard: tensorboard --logdir logs")
    print("View at: http://localhost:6006\n")

    # Create runner
    runner = OnPolicyRunnerWithFlush(
        env=env,
        train_cfg=train_cfg_dict,
        log_dir=log_dir,
        device=env.device,
    )

    # Train
    print("Starting training...\n")
    runner.learn(
        num_learning_iterations=train_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    print("\nTraining complete!")
    print(f"Logs saved to: {log_dir}")
    print(f"View with: tensorboard --logdir {log_dir}\n")

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
