"""
Training script for quadruped locomotion with Isaac Lab and RSL-RL.
"""
import os

os.environ["PYTORCH_NVFUSER_DISABLE_FALLBACK"] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

HEADLESS = True

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

import pickle
import shutil

import torch
from isaaclab.envs import ManagerBasedRLEnv

from rsl_rl.runners import OnPolicyRunner

from rl_cfg import QuadrupedEnvCfg
from train_cfg import TrainCfg, RewardWeightsCfg
from quadruped_env import IsaacLabVecEnvWrapper


def apply_reward_weights(env, reward_weights: RewardWeightsCfg):
    """Apply reward weights to the environment."""
    isaac_env = env.unwrapped
    
    if not hasattr(isaac_env, 'reward_manager'):
        return
    
    reward_manager = isaac_env.reward_manager
    
    weight_map = {
        "joint_jerk": reward_weights.efficiency_weight * reward_weights.joint_jerk,
        "joint_torque": reward_weights.efficiency_weight * reward_weights.joint_torque,
        "action_smoothness": reward_weights.efficiency_weight * reward_weights.action_smoothness,
        "velocity_tracking": reward_weights.velocity_weight * reward_weights.velocity_tracking,
        "foot_slip": reward_weights.stability_weight * reward_weights.foot_slip,
        "body_rates": reward_weights.stability_weight * reward_weights.body_rates,
        "base_orientation": reward_weights.stability_weight * reward_weights.base_orientation,
        "ride_height": reward_weights.stability_weight * reward_weights.ride_height,
        "hip_position": reward_weights.stability_weight * reward_weights.hip_position,
        "leg_collision": reward_weights.stability_weight * reward_weights.leg_collision,
    }
    
    for reward_name, weight in weight_map.items():
        if hasattr(reward_manager, '_term_names') and hasattr(reward_manager, '_term_cfgs'):
            if reward_name in reward_manager._term_names:
                idx = reward_manager._term_names.index(reward_name)
                if isinstance(reward_manager._term_cfgs, list):
                    reward_manager._term_cfgs[idx].weight = weight
                else:
                    reward_manager._term_cfgs[reward_name].weight = weight


class GaitRewardWrapper(IsaacLabVecEnvWrapper):
    """Wrapper that adds gait tracking rewards to the environment rewards."""
    
    REWARD_CATEGORIES = {
        "Efficiency": ["joint_jerk", "joint_torque", "action_smoothness"],
        "Velocity": ["velocity_tracking"],
        "Gait_Tracking": ["gait_foot_contact", "gait_step_height"],
        "Stability": ["foot_slip", "body_rates", "base_orientation", "ride_height", "hip_position", "leg_collision"],
    }
    
    def __init__(self, env, reward_weights: RewardWeightsCfg, sensor_cfg=None):
        self.reward_weights = reward_weights
        self._episode_sums = None
        self._prev_isaac_episode_sums = None
        super().__init__(env, sensor_cfg=sensor_cfg)
        
        self._episode_sums = {
            "gait_foot_contact": torch.zeros(self.num_envs, device=self.device),
            "gait_step_height": torch.zeros(self.num_envs, device=self.device),
        }
        
        self._prev_isaac_episode_sums = {}
    
    def reset(self):
        """Reset and clear episode sums."""
        obs = super().reset()
        if self._episode_sums is not None:
            for key in self._episode_sums:
                self._episode_sums[key].zero_()
        self._prev_isaac_episode_sums = {}
        return obs
    
    def _get_category_for_reward(self, reward_name: str) -> str:
        """Get the category for a given reward name."""
        for category, rewards in self.REWARD_CATEGORIES.items():
            if reward_name in rewards:
                return category
        return "Other"
    
    def _reorganize_reward_logs(self, extras: dict) -> dict:
        """Reorganize reward logs from Episode_Reward/* to Category/reward_name format."""
        if "log" not in extras:
            return extras
        
        log = extras["log"]
        new_log = {}
        category_sums = {}
        
        for key, value in log.items():
            if key.startswith("Episode_Reward/"):
                reward_name = key.replace("Episode_Reward/", "")
                category = self._get_category_for_reward(reward_name)
                
                new_key = f"{category}/{reward_name}"
                new_log[new_key] = value
                
                if category not in category_sums:
                    category_sums[category] = 0.0
                category_sums[category] += value
            else:
                new_log[key] = value
        
        for category, total in category_sums.items():
            new_log[f"Category_Total/{category}"] = total
        
        extras["log"] = new_log
        return extras
    
    def step(self, actions):
        """Step with added gait rewards."""
        obs, rewards, dones, extras = super().step(actions)
        
        gait_rewards, gait_components = self._compute_gait_rewards()
        rewards = rewards + gait_rewards
        
        self._episode_sums["gait_foot_contact"] += gait_components["foot_contact"]
        self._episode_sums["gait_step_height"] += gait_components["step_height"]
        
        if "log" not in extras:
            extras["log"] = {}
        
        num_dones = dones.sum().item()
        if num_dones > 0:
            done_mask = dones.float()
            
            extras["log"]["Episode_Reward/gait_foot_contact"] = (self._episode_sums["gait_foot_contact"] * done_mask).sum().item() / max(num_dones, 1)
            extras["log"]["Episode_Reward/gait_step_height"] = (self._episode_sums["gait_step_height"] * done_mask).sum().item() / max(num_dones, 1)
            
            for key in self._episode_sums:
                self._episode_sums[key] = torch.where(
                    dones, 
                    torch.zeros_like(self._episode_sums[key]),
                    self._episode_sums[key]
                )
        
        extras = self._reorganize_reward_logs(extras)
        
        return obs, rewards, dones, extras
    
    def _compute_gait_rewards(self) -> tuple[torch.Tensor, dict]:
        """Compute gait scheduler tracking rewards."""
        gait_scheduler = self.gait_scheduler
        isaac_env = self._env
        
        desired_contacts, desired_heights = gait_scheduler.get_gait_obs()
        
        foot_contact_sensor = isaac_env.scene.sensors["foot_contact"]
        contact_forces = foot_contact_sensor.data.net_forces_w
        contact_threshold = self.sensor_cfg.foot_contact_threshold
        actual_contacts = (torch.norm(contact_forces, dim=-1) > contact_threshold).float()
        
        contact_match = 1.0 - torch.abs(desired_contacts - actual_contacts)
        contact_reward = torch.mean(contact_match, dim=-1)
        
        robot = isaac_env.scene["robot"]
        body_pos = robot.data.body_pos_w
        num_feet = 4
        
        if body_pos.shape[1] >= num_feet:
            foot_pos_z = body_pos[:, -num_feet:, 2]
        else:
            foot_pos_z = body_pos[:, :num_feet, 2]
        
        terrain_height = torch.zeros(self.num_envs, device=self.device)
        
        foot_heights_above_ground = foot_pos_z - terrain_height.unsqueeze(-1)
        foot_heights_above_ground = torch.clamp(foot_heights_above_ground, min=0.0)
        
        height_error = torch.abs(foot_heights_above_ground - desired_heights)
        height_reward = torch.exp(-height_error.pow(2) / (0.02 * 0.02))
        height_reward = torch.mean(height_reward, dim=-1)
        
        gait_weight = self.reward_weights.gait_tracking_weight
        contact_weight = self.reward_weights.foot_contact_tracking
        step_height_weight = self.reward_weights.step_height_tracking
        
        weighted_contact = gait_weight * contact_weight * contact_reward
        weighted_height = gait_weight * step_height_weight * height_reward
        
        total_reward = weighted_contact + weighted_height
        
        reward_components = {
            "foot_contact": weighted_contact,
            "step_height": weighted_height,
        }
        
        return total_reward, reward_components


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

    env_cfg = QuadrupedEnvCfg()
    train_cfg = TrainCfg()

    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    
    env = GaitRewardWrapper(isaac_env, train_cfg.reward_weights, sensor_cfg=train_cfg.sensor)

    apply_reward_weights(env, train_cfg.reward_weights)

    log_dir = f"logs/{train_cfg.experiment_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump({
        "env_cfg": env_cfg,
        "train_cfg": train_cfg,
    }, open(f"{log_dir}/cfgs.pkl", "wb"))

    train_cfg_dict = get_rsl_rl_cfg(train_cfg, env_cfg.scene.num_envs)

    print("\n" + "=" * 80)
    print("QUADRUPED RL TRAINING (Gait Scheduler)")
    print("=" * 80)
    print(f"Experiment: {train_cfg.experiment_name}")
    print(f"Max iterations: {train_cfg.max_iterations}")
    print(f"Num environments: {env_cfg.scene.num_envs}")
    print(f"Steps per env: {train_cfg.num_steps_per_env}")
    print(f"Log directory: {log_dir}")
    print(f"Device: {env.device}")
    print(f"Network: {train_cfg.policy['actor_hidden_dims']}")
    print(f"Activation: {train_cfg.policy['activation']}")
    print(f"Observation dim: {env.num_obs}")
    print("=" * 80)
    print("\nReward Group Weights:")
    rw = train_cfg.reward_weights
    print(f"  Efficiency: {rw.efficiency_weight}")
    print(f"  Velocity: {rw.velocity_weight}")
    print(f"  Gait Tracking: {rw.gait_tracking_weight}")
    print(f"  Stability: {rw.stability_weight}")
    print("=" * 80)
    print("\nTensorBoard: tensorboard --logdir logs")
    print("View at: http://localhost:6006\n")

    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg_dict,
        log_dir=log_dir,
        device=env.device,
    )

    print("Starting training...\n")
    runner.learn(
        num_learning_iterations=train_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    print("\nTraining complete!")
    print(f"Logs saved to: {log_dir}")
    print(f"View with: tensorboard --logdir {log_dir}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()