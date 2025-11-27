"""
Training script for quadruped locomotion with RSL-RL.
This version FIXES the TensorBoard live update issue by adding explicit flushes.
"""
import os
import pickle
import shutil

import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from quadruped_env import QuadrupedEnv
from configs import (
    EnvConfig, 
    ObsConfig, 
    RewardConfig, 
    CommandConfig, 
    TrainConfig,
    EvalConfig,
    get_cfg_dict
)


class OnPolicyRunnerWithFlush(OnPolicyRunner):
    """
    OnPolicyRunner with explicit TensorBoard flush after each log.
    This fixes the issue where TensorBoard doesn't update during training.
    """
    
    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Override log method to add explicit flush."""
        # Call parent log method (this handles all the infos["log"] entries automatically)
        super().log(locs, width, pad)
        
        # CRITICAL FIX: Explicitly flush the writer after logging
        # This ensures TensorBoard updates during training, not just at the end
        if self.writer is not None:
            self.writer.flush()
    
    def run(self, num_learning_iterations: int, init_at_random_ep_len: bool = True):
        """Override to capture extras during rollout."""
        # Wrap env.step to capture reward components
        original_step = self.env.step
        reward_sums = {}
        reward_counts = 0
        
        def step_wrapper(actions):
            nonlocal reward_counts
            obs, rewards, dones, extras = original_step(actions)
            # Accumulate reward components
            for key, value in extras.items():
                if key.startswith('reward_'):
                    if key not in reward_sums:
                        reward_sums[key] = 0.0
                    reward_sums[key] += float(value)
            reward_counts += 1
            return obs, rewards, dones, extras
        
        # Temporarily replace step
        self.env.step = step_wrapper
        
        # Run parent's rollout and training
        result = super().run(num_learning_iterations, init_at_random_ep_len)
        
        # Calculate averages and store for logging
        if reward_counts > 0:
            self.episode_reward_components = {
                key: value / reward_counts for key, value in reward_sums.items()
            }
        
        # Restore original step
        self.env.step = original_step
        
        return result


def train():
    """Main training function."""
    
    # Load configurations (edit configs.py to change these)
    env_cfg = EnvConfig()
    obs_cfg = ObsConfig()
    reward_cfg = RewardConfig()
    command_cfg = CommandConfig()
    train_cfg = TrainConfig()
    eval_cfg = EvalConfig()
    
    # Initialize Genesis
    gs.init(logging_level="warning")
    
    # Create environment
    env = QuadrupedEnv(
        num_envs=env_cfg.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )
    
    # Setup logging (automatically overwrites old logs)
    log_dir = f"logs/{train_cfg.experiment_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configurations
    pickle.dump({
        'env_cfg': env_cfg,
        'obs_cfg': obs_cfg,
        'reward_cfg': reward_cfg,
        'command_cfg': command_cfg,
        'train_cfg': train_cfg,
        'eval_cfg': eval_cfg,
    }, open(f"{log_dir}/cfgs.pkl", "wb"))
    
    # Get RSL-RL config dict
    train_cfg_dict = get_cfg_dict(train_cfg, env_cfg)
    
    # Print training info
    print("\n" + "="*80)
    print("QUADRUPED RL TRAINING (with TensorBoard fix)")
    print("="*80)
    print(f"Experiment: {train_cfg.experiment_name}")
    print(f"Max iterations: {train_cfg.max_iterations}")
    print(f"Num environments: {env_cfg.num_envs}")
    print(f"Steps per env: {train_cfg.num_steps_per_env}")
    print(f"Total steps per iteration: {train_cfg.num_steps_per_env * env_cfg.num_envs}")
    print(f"Log directory: {log_dir}")
    print(f"Device: {env.device}")
    print("="*80)
    print("\nTensorBoard will now update LIVE during training!")
    print("   Open in another terminal: tensorboard --logdir logs")
    print("   Then go to: http://localhost:6006")
    print("   Refresh (Ctrl+R) if needed\n")
    
    # Create runner with flush fix
    runner = OnPolicyRunnerWithFlush(
        env=env,
        train_cfg=train_cfg_dict,
        log_dir=log_dir,
        device=env.device
    )
    
    # Train
    print("Starting training...\n")
    runner.learn(
        num_learning_iterations=train_cfg.max_iterations,
        init_at_random_ep_len=True
    )
    
    print("\nTraining complete!")
    print(f"Logs saved to: {log_dir}")
    print(f"View with: tensorboard --logdir {log_dir}\n")


if __name__ == "__main__":
    train()
