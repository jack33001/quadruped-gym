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

# Configuration - set this before AppLauncher
HEADLESS = True  # Set to False to show viewer for debugging

from isaaclab.app import AppLauncher

# Create app launcher and launch the simulator
app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

# Now we can import everything else
import pickle
import shutil

import torch
from isaaclab.envs import ManagerBasedRLEnv

from rsl_rl.runners import OnPolicyRunner

from rl_cfg import QuadrupedEnvCfg, TrainCfg
from quadruped_env import IsaacLabVecEnvWrapper


def apply_curriculum_stage(env, train_cfg: TrainCfg, stage: int, verbose: bool = True):
    """Apply reward weights and commands for a given curriculum stage."""
    isaac_env = env.unwrapped
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"CURRICULUM STAGE {stage}")
        print(f"{'='*60}")
    
    # Update reward weights
    for reward_name, weights in train_cfg.curriculum_rewards.items():
        if hasattr(isaac_env, 'reward_manager'):
            reward_manager = isaac_env.reward_manager
            if hasattr(reward_manager, '_term_names') and hasattr(reward_manager, '_term_cfgs'):
                if reward_name in reward_manager._term_names:
                    idx = reward_manager._term_names.index(reward_name)
                    if isinstance(reward_manager._term_cfgs, list):
                        reward_manager._term_cfgs[idx].weight = weights[stage]
                    else:
                        reward_manager._term_cfgs[reward_name].weight = weights[stage]
                    if verbose:
                        print(f"  {reward_name}: weight = {weights[stage]}")
    
    # Update command ranges
    cmd_ranges = train_cfg.curriculum_commands
    if hasattr(isaac_env, 'command_manager'):
        for term in isaac_env.command_manager._terms.values():
            if hasattr(term, 'cfg') and hasattr(term.cfg, 'ranges'):
                term.cfg.ranges.lin_vel_x = cmd_ranges["lin_vel_x"][stage]
                term.cfg.ranges.lin_vel_y = cmd_ranges["lin_vel_y"][stage]
                term.cfg.ranges.ang_vel_z = cmd_ranges["ang_vel_z"][stage]
                if verbose:
                    print(f"  Commands: lin_vel_x={cmd_ranges['lin_vel_x'][stage]}, "
                          f"lin_vel_y={cmd_ranges['lin_vel_y'][stage]}, "
                          f"ang_vel_z={cmd_ranges['ang_vel_z'][stage]}")
    
    # Update terrain level
    terrain_level = env.set_terrain_level_for_stage(stage, train_cfg.curriculum_num_stages)
    if verbose:
        print(f"  Terrain level: {terrain_level} / {env.max_terrain_level}")
        print(f"{'='*60}\n")


class OnPolicyRunnerWithCurriculum(OnPolicyRunner):
    """
    OnPolicyRunner with unified performance-based curriculum.
    """
    
    def __init__(self, *args, train_cfg_obj=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_cfg_obj = train_cfg_obj
        self.current_stage = 0
        self._curriculum_update_counter = 0
        self._stage_episode_rewards = []  # Track rewards at current stage
        self._cooldown_remaining = 0  # Iterations to wait after stage change

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Override log method to add unified curriculum update."""
        if self.train_cfg_obj is not None:
            self._curriculum_update_counter += 1
            
            # Collect completed episodes
            completed = self.env.get_completed_episodes()
            self._stage_episode_rewards.extend([r for r, _ in completed])
            
            # Check curriculum update
            if self._curriculum_update_counter >= self.train_cfg_obj.curriculum_update_freq:
                self._curriculum_update_counter = 0
                self._update_unified_curriculum()
        
        # Log terrain level distribution
        if hasattr(self.env, 'get_terrain_levels'):
            terrain_levels = self.env.get_terrain_levels()
        elif hasattr(self.env, 'terrain_levels'):
            terrain_levels = self.env.terrain_levels
        else:
            terrain_levels = None
        
        if terrain_levels is not None and self.writer is not None:
            mean_level = terrain_levels.float().mean().item()
            self.writer.add_scalar(
                "Curriculum/terrain_level", 
                mean_level, 
                self.current_learning_iteration
            )
            self.writer.add_scalar(
                "Curriculum/stage", 
                self.current_stage, 
                self.current_learning_iteration
            )
            
            # Log mean episode reward for curriculum
            if len(self._stage_episode_rewards) > 0:
                mean_reward = sum(self._stage_episode_rewards) / len(self._stage_episode_rewards)
                self.writer.add_scalar(
                    "Curriculum/mean_episode_reward",
                    mean_reward,
                    self.current_learning_iteration
                )
        
        super().log(locs, width, pad)
        if self.writer is not None:
            self.writer.flush()

    def _update_unified_curriculum(self):
        """Update curriculum stage based on recent episode performance."""
        cfg = self.train_cfg_obj
        
        # Check if we're in cooldown period after a stage change
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            # Still collect episodes but don't make decisions
            if self._cooldown_remaining == 0:
                # Cooldown finished, clear old data to start fresh evaluation
                self._stage_episode_rewards.clear()
                print(f"  [Curriculum] Cooldown finished, collecting fresh data for stage {self.current_stage}")
            return
        
        # Need minimum episodes to make a decision
        if len(self._stage_episode_rewards) < cfg.curriculum_min_episodes:
            return
        
        mean_reward = sum(self._stage_episode_rewards) / len(self._stage_episode_rewards)
        
        old_stage = self.current_stage
        
        # Promote if doing well
        if mean_reward >= cfg.curriculum_promote_threshold:
            if self.current_stage < cfg.curriculum_num_stages - 1:
                self.current_stage += 1
                self._stage_episode_rewards.clear()  # Reset for new stage
                self._cooldown_remaining = cfg.curriculum_cooldown_promote
                apply_curriculum_stage(self.env, cfg, self.current_stage)
                print(f"  [Curriculum] Promoted to stage {self.current_stage}! "
                      f"Mean reward {mean_reward:.1f} >= {cfg.curriculum_promote_threshold}")
                print(f"  [Curriculum] Entering cooldown for {self._cooldown_remaining} iterations")
                return
        
        # Demote if struggling
        elif mean_reward < cfg.curriculum_demote_threshold:
            if self.current_stage > 0:
                self.current_stage -= 1
                self._stage_episode_rewards.clear()  # Reset for new stage
                self._cooldown_remaining = cfg.curriculum_cooldown_demote
                apply_curriculum_stage(self.env, cfg, self.current_stage)
                print(f"  [Curriculum] Demoted to stage {self.current_stage}. "
                      f"Mean reward {mean_reward:.1f} < {cfg.curriculum_demote_threshold}")
                print(f"  [Curriculum] Entering cooldown for {self._cooldown_remaining} iterations")
                return
        
        # If stage didn't change but we have enough data, clear old data to keep window fresh
        if self.current_stage == old_stage and len(self._stage_episode_rewards) > cfg.curriculum_min_episodes * 2:
            # Keep only recent half
            self._stage_episode_rewards = self._stage_episode_rewards[-cfg.curriculum_min_episodes:]

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

    # Apply initial curriculum stage (stage 0)
    apply_curriculum_stage(env, train_cfg, stage=0, verbose=True)

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
    print(f"Curriculum: {train_cfg.curriculum_num_stages} stages (unified terrain + rewards)")
    print(f"  Promote threshold: {train_cfg.curriculum_promote_threshold}")
    print(f"  Demote threshold: {train_cfg.curriculum_demote_threshold}")
    print("=" * 80)
    print("\nTensorBoard: tensorboard --logdir logs")
    print("View at: http://localhost:6006\n")

    # Create runner with curriculum support
    runner = OnPolicyRunnerWithCurriculum(
        env=env,
        train_cfg=train_cfg_dict,
        log_dir=log_dir,
        device=env.device,
        train_cfg_obj=train_cfg,
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
