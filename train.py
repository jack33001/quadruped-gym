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
from train_cfg import TrainCfg, RewardWeightsCfg, VelocityCurriculumCfg
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
        "Gait_Tracking": ["gait_foot_contact", "gait_foot_trajectory"],
        "Stability": ["foot_slip", "body_rates", "base_orientation", "ride_height", "hip_position", "leg_collision"],
    }
    
    def __init__(self, env, reward_weights: RewardWeightsCfg, sensor_cfg=None):
        self.reward_weights = reward_weights
        self._episode_sums = None
        self._prev_isaac_episode_sums = None
        self._velocity_error_sum = None
        self._velocity_error_count = None
        self._velocity_curriculum = None
        self._step_count = 0
        self._current_iteration = 0
        
        super().__init__(env, sensor_cfg=sensor_cfg)
        
        self._episode_sums = {
            "gait_foot_contact": torch.zeros(self.num_envs, device=self.device),
            "gait_foot_trajectory": torch.zeros(self.num_envs, device=self.device),
        }
        
        self._prev_isaac_episode_sums = {}
        
        self._velocity_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._velocity_error_count = torch.zeros(self.num_envs, device=self.device)
    
    def set_velocity_curriculum(self, curriculum):
        """Set the velocity curriculum manager."""
        self._velocity_curriculum = curriculum
    
    def set_iteration(self, iteration: int):
        """Update current iteration for curriculum."""
        if self._current_iteration != iteration:
            self._current_iteration = iteration
            if self._velocity_curriculum is not None:
                self._velocity_curriculum.update(iteration)
    
    def reset(self):
        """Reset and clear episode sums."""
        obs = super().reset()
        if self._episode_sums is not None:
            for key in self._episode_sums:
                self._episode_sums[key].zero_()
        self._prev_isaac_episode_sums = {}
        if self._velocity_error_sum is not None:
            self._velocity_error_sum.zero_()
        if self._velocity_error_count is not None:
            self._velocity_error_count.zero_()
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
        self._episode_sums["gait_foot_trajectory"] += gait_components["foot_trajectory"]
        
        velocity_error = self._compute_velocity_error()
        self._velocity_error_sum += velocity_error
        self._velocity_error_count += 1
        
        if "log" not in extras:
            extras["log"] = {}
        
        num_dones = dones.sum().item()
        if num_dones > 0:
            done_mask = dones.float()
            
            extras["log"]["Episode_Reward/gait_foot_contact"] = (self._episode_sums["gait_foot_contact"] * done_mask).sum().item() / max(num_dones, 1)
            extras["log"]["Episode_Reward/gait_foot_trajectory"] = (self._episode_sums["gait_foot_trajectory"] * done_mask).sum().item() / max(num_dones, 1)
            
            avg_vel_error = self._velocity_error_sum / self._velocity_error_count.clamp(min=1)
            extras["log"]["Metrics/avg_velocity_error"] = (avg_vel_error * done_mask).sum().item() / max(num_dones, 1)
            
            for key in self._episode_sums:
                self._episode_sums[key] = torch.where(
                    dones, 
                    torch.zeros_like(self._episode_sums[key]),
                    self._episode_sums[key]
                )
            
            self._velocity_error_sum = torch.where(dones, torch.zeros_like(self._velocity_error_sum), self._velocity_error_sum)
            self._velocity_error_count = torch.where(dones, torch.zeros_like(self._velocity_error_count), self._velocity_error_count)
        
        extras = self._reorganize_reward_logs(extras)
        
        return obs, rewards, dones, extras
    
    def _compute_gait_rewards(self) -> tuple[torch.Tensor, dict]:
        """Compute gait scheduler tracking rewards."""
        gait_scheduler = self.gait_scheduler
        isaac_env = self._env
        
        desired_contacts, desired_foot_xy, desired_heights = gait_scheduler.get_gait_obs()
        
        foot_contact_sensor = isaac_env.scene.sensors["foot_contact"]
        contact_forces = foot_contact_sensor.data.net_forces_w
        contact_threshold = self.sensor_cfg.foot_contact_threshold
        actual_contacts = (torch.norm(contact_forces, dim=-1) > contact_threshold).float()
        
        contact_match = 1.0 - torch.abs(desired_contacts - actual_contacts)
        contact_reward = torch.mean(contact_match, dim=-1)
        
        robot = isaac_env.scene["robot"]
        body_pos = robot.data.body_pos_w
        root_pos = robot.data.root_pos_w
        root_quat = robot.data.root_quat_w
        num_feet = 4
        
        if body_pos.shape[1] >= num_feet:
            foot_pos_world = body_pos[:, -num_feet:, :]
        else:
            foot_pos_world = body_pos[:, :num_feet, :]
        
        foot_pos_local = self._world_to_body_frame(foot_pos_world, root_pos, root_quat)
        
        terrain_height = torch.zeros(self.num_envs, device=self.device)
        if hasattr(isaac_env.scene, 'terrain') and isaac_env.scene.terrain is not None:
            terrain = isaac_env.scene.terrain
            if hasattr(terrain, 'env_origins'):
                terrain_height = terrain.env_origins[:, 2]
        
        foot_z_local = foot_pos_world[:, :, 2] - terrain_height.unsqueeze(-1)
        foot_z_local = torch.clamp(foot_z_local, min=0.0)
        
        xy_error = foot_pos_local[:, :, :2] - desired_foot_xy
        xy_error_sq = (xy_error ** 2).sum(dim=-1)
        
        z_error = foot_z_local - desired_heights
        z_error_sq = z_error ** 2
        
        total_error_sq = xy_error_sq + z_error_sq
        trajectory_reward = torch.exp(-total_error_sq / (0.02 * 0.02))
        trajectory_reward = torch.mean(trajectory_reward, dim=-1)
        
        gait_weight = self.reward_weights.gait_tracking_weight
        contact_weight = self.reward_weights.foot_contact_tracking
        trajectory_weight = self.reward_weights.foot_trajectory_tracking
        
        weighted_contact = gait_weight * contact_weight * contact_reward
        weighted_trajectory = gait_weight * trajectory_weight * trajectory_reward
        
        total_reward = weighted_contact + weighted_trajectory
        
        reward_components = {
            "foot_contact": weighted_contact,
            "foot_trajectory": weighted_trajectory,
        }
        
        return total_reward, reward_components
    
    def _compute_velocity_error(self) -> torch.Tensor:
        """Compute velocity tracking error as percentage of commanded velocity."""
        isaac_env = self._env
        robot = isaac_env.scene["robot"]
        cmd_manager = isaac_env.command_manager
        
        actual_vel = robot.data.root_lin_vel_w[:, :2]
        
        if hasattr(cmd_manager, 'get_command'):
            velocity_cmd = cmd_manager.get_command("base_velocity")
            if velocity_cmd is not None:
                commanded_vel = velocity_cmd[:, :2]
                commanded_magnitude = torch.norm(commanded_vel, dim=-1)
                error_magnitude = torch.norm(commanded_vel - actual_vel, dim=-1)
                
                percent_error = torch.where(
                    commanded_magnitude > 0.01,
                    100.0 * error_magnitude / commanded_magnitude,
                    torch.zeros_like(error_magnitude)
                )
                return percent_error
        
        return torch.zeros(self.num_envs, device=self.device)
    
    def _world_to_body_frame(self, points_world: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
        """Transform points from world frame to body frame.
        
        Args:
            points_world: Points in world frame (num_envs, num_points, 3)
            root_pos: Root position (num_envs, 3)
            root_quat: Root quaternion (num_envs, 4) in (w, x, y, z) order
            
        Returns:
            Points in body frame (num_envs, num_points, 3)
        """
        points_rel = points_world - root_pos.unsqueeze(1)
        
        w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        
        inv_w = w
        inv_x = -x
        inv_y = -y
        inv_z = -z
        
        num_points = points_rel.shape[1]
        points_body = torch.zeros_like(points_rel)
        
        for i in range(num_points):
            px, py, pz = points_rel[:, i, 0], points_rel[:, i, 1], points_rel[:, i, 2]
            
            t0 = inv_w * px + inv_y * pz - inv_z * py
            t1 = inv_w * py + inv_z * px - inv_x * pz
            t2 = inv_w * pz + inv_x * py - inv_y * px
            t3 = -inv_x * px - inv_y * py - inv_z * pz
            
            points_body[:, i, 0] = t0 * inv_w - t3 * inv_x - t1 * inv_z + t2 * inv_y
            points_body[:, i, 1] = t1 * inv_w - t3 * inv_y - t2 * inv_x + t0 * inv_z
            points_body[:, i, 2] = t2 * inv_w - t3 * inv_z - t0 * inv_y + t1 * inv_x
        
        return points_body

class VelocityCurriculum:
    """Manages velocity command curriculum during training."""
    
    def __init__(self, cfg: VelocityCurriculumCfg, isaac_env):
        self.cfg = cfg
        self.isaac_env = isaac_env
        self.current_stage = 0
        self._last_vel_range = None
        self._iteration = 0
    
    def get_vel_range_for_iteration(self, iteration: int) -> tuple:
        """Get the velocity range for the current iteration."""
        for stage in self.cfg.stages:
            if iteration < stage["end_iteration"]:
                return stage["vel_range"]
        return self.cfg.stages[-1]["vel_range"]
    
    def update(self, iteration: int):
        """Update velocity command range based on current iteration."""
        self._iteration = iteration
        vel_range = self.get_vel_range_for_iteration(iteration)
        
        if vel_range != self._last_vel_range:
            self._apply_vel_range(vel_range)
            self._last_vel_range = vel_range
            print(f"[Curriculum] Iteration {iteration}: velocity range set to {vel_range}")
    
    def _apply_vel_range(self, vel_range: tuple):
        """Apply velocity range to environment command manager."""
        cmd_manager = self.isaac_env.command_manager
        
        if hasattr(cmd_manager, '_terms') and 'base_velocity' in cmd_manager._terms:
            term = cmd_manager._terms['base_velocity']
            if hasattr(term, 'cfg') and hasattr(term.cfg, 'ranges'):
                term.cfg.ranges.lin_vel_x = vel_range


class CurriculumGaitRewardWrapper(GaitRewardWrapper):
    """Wrapper that adds curriculum support via step counting."""
    
    def __init__(self, env, reward_weights: RewardWeightsCfg, sensor_cfg=None, 
                 curriculum: VelocityCurriculum = None, steps_per_iteration: int = 24):
        super().__init__(env, reward_weights, sensor_cfg)
        self._curriculum = curriculum
        self._steps_per_iteration = steps_per_iteration * self.num_envs
        self._total_steps = 0
        self._last_iteration = -1
    
    def step(self, actions):
        self._total_steps += self.num_envs
        
        current_iteration = self._total_steps // self._steps_per_iteration
        if current_iteration != self._last_iteration:
            self._last_iteration = current_iteration
            if self._curriculum is not None:
                self._curriculum.update(current_iteration)
        
        return super().step(actions)


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
    
    velocity_curriculum = VelocityCurriculum(train_cfg.velocity_curriculum, isaac_env)
    
    env = CurriculumGaitRewardWrapper(
        isaac_env, 
        train_cfg.reward_weights, 
        sensor_cfg=train_cfg.sensor,
        curriculum=velocity_curriculum,
        steps_per_iteration=train_cfg.num_steps_per_env
    )

    apply_reward_weights(env, train_cfg.reward_weights)
    velocity_curriculum.update(0)

    log_dir = f"logs/{train_cfg.experiment_name}"
    models_dir = f"{log_dir}/models"
    
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

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
    print("\nVelocity Curriculum:")
    for stage in train_cfg.velocity_curriculum.stages:
        print(f"  Until iteration {stage['end_iteration']}: {stage['vel_range']}")
    print("=" * 80)
    print("\nTensorBoard: tensorboard --logdir logs")
    print("View at: http://localhost:6006\n")

    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg_dict,
        log_dir=models_dir,
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