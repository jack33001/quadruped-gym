"""
RSL-RL compatible wrapper for Isaac Lab quadruped environment.
"""
import math
import torch
from tensordict import TensorDict

from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.env import VecEnv

from gait_cfg import GaitScheduler, GaitSchedulerCfg


class IsaacLabVecEnvWrapper(VecEnv):
    """
    Wrapper to make Isaac Lab ManagerBasedRLEnv compatible with RSL-RL.
    """

    def __init__(self, env: ManagerBasedRLEnv, sensor_cfg=None):
        """Initialize wrapper."""
        self._env = env
        
        if sensor_cfg is None:
            from train_cfg import SensorCfg
            sensor_cfg = SensorCfg()
        self.sensor_cfg = sensor_cfg
        
        # VecEnv required attributes
        self.num_envs = env.num_envs
        self.num_privileged_obs = None
        self.num_actions = env.action_manager.total_action_dim
        self.max_episode_length = int(env.max_episode_length)
        self.device = env.device

        # Buffers
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Gait scheduler
        gait_cfg = GaitSchedulerCfg()
        self.gait_scheduler = GaitScheduler(self.num_envs, self.device, gait_cfg)
        self.sim_dt = env.step_dt
        
        # Phase oscillator synced to gait frequency
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)
        
        # Previous state buffers (for observation)
        # Will be initialized on first observation
        self._prev_imu_ang_vel = None
        self._prev_imu_projected_gravity = None
        self._prev_joint_pos = None
        self._prev_joint_vel = None
        
        # Store last observation for get_observations()
        self._last_obs = None
        
        # Calculate observation dimension
        # Base: commands(3) + projected_gravity(3) + ang_vel(3) + joint_pos(8) + joint_vel(8) + last_action(8) = 33
        # Prev state: prev_ang_vel(3) + prev_projected_gravity(3) + prev_joint_pos(8) + prev_joint_vel(8) = 22
        # Gait: contact_states(4) + foot_heights(4) = 8
        # Phase oscillator: sin(phase) + cos(phase) = 2
        # Total additional: 32
        base_obs_dim = env.observation_manager.group_obs_dim["policy"][0]
        self.num_obs = base_obs_dim + 22 + 8 + 2
        
        # Debug counters
        self._nan_obs_count = 0
        self._nan_reward_count = 0
        
        # Completed episodes buffer for curriculum
        self._completed_episodes = []
        
        # Episode tracking
        self.episode_rewards_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_steps = torch.zeros(self.num_envs, device=self.device)
        
        # Terrain curriculum tracking
        self._setup_terrain_curriculum()
        
        # Do initial reset to populate observations
        self.reset()

    def _setup_terrain_curriculum(self):
        """Setup terrain curriculum tracking buffers."""
        self.terrain_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.num_terrain_rows = 5
        self.num_terrain_cols = 5
        self.max_terrain_level = self.num_terrain_rows - 1
        
        if hasattr(self._env, 'scene') and hasattr(self._env.scene, 'terrain'):
            terrain = self._env.scene.terrain
            if hasattr(terrain, 'cfg') and hasattr(terrain.cfg, 'terrain_generator'):
                gen_cfg = terrain.cfg.terrain_generator
                if hasattr(gen_cfg, 'num_rows'):
                    self.num_terrain_rows = gen_cfg.num_rows
                    self.max_terrain_level = self.num_terrain_rows - 1
                if hasattr(gen_cfg, 'num_cols'):
                    self.num_terrain_cols = gen_cfg.num_cols

    def set_terrain_level_for_stage(self, stage: int, num_stages: int) -> int:
        """Set terrain level based on curriculum stage."""
        if num_stages <= 1:
            target_level = 0
        else:
            target_level = int((stage / (num_stages - 1)) * self.max_terrain_level)
        
        target_level = min(target_level, self.max_terrain_level)
        self.terrain_levels.fill_(target_level)
        
        if hasattr(self._env, 'scene') and hasattr(self._env.scene, 'terrain'):
            terrain = self._env.scene.terrain
            if hasattr(terrain, 'cfg'):
                terrain.cfg.max_init_terrain_level = target_level
        
        return target_level

    def get_terrain_levels(self) -> torch.Tensor:
        """Get current terrain levels for all environments."""
        return self.terrain_levels

    def get_completed_episodes(self):
        """Get list of completed episodes with their rewards and lengths."""
        completed = self._completed_episodes.copy()
        self._completed_episodes.clear()
        return completed

    def _get_robot_state(self):
        """Extract current robot state for observation and previous state tracking."""
        # Get IMU data
        imu_sensor = self._env.scene.sensors["imu"]
        ang_vel = imu_sensor.data.ang_vel_b.clone()
        
        # Compute projected gravity from IMU orientation
        quat = imu_sensor.data.quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        gx = 2 * (x * z - w * y)
        gy = 2 * (y * z + w * x)
        gz = w * w - x * x - y * y + z * z
        projected_gravity = torch.stack([gx, gy, gz], dim=-1)
        
        # Get joint state
        robot = self._env.scene["robot"]
        joint_pos = (robot.data.joint_pos - robot.data.default_joint_pos).clone()
        joint_vel = robot.data.joint_vel.clone()
        
        return ang_vel, projected_gravity, joint_pos, joint_vel

    def _init_prev_state(self, ang_vel, projected_gravity, joint_pos, joint_vel):
        """Initialize previous state buffers with current state."""
        self._prev_imu_ang_vel = ang_vel.clone()
        self._prev_imu_projected_gravity = projected_gravity.clone()
        self._prev_joint_pos = joint_pos.clone()
        self._prev_joint_vel = joint_vel.clone()

    def _update_prev_state(self, ang_vel, projected_gravity, joint_pos, joint_vel, reset_ids=None):
        """Update previous state buffers."""
        if reset_ids is not None and len(reset_ids) > 0:
            self._prev_imu_ang_vel[reset_ids] = ang_vel[reset_ids]
            self._prev_imu_projected_gravity[reset_ids] = projected_gravity[reset_ids]
            self._prev_joint_pos[reset_ids] = joint_pos[reset_ids]
            self._prev_joint_vel[reset_ids] = joint_vel[reset_ids]
        else:
            self._prev_imu_ang_vel = ang_vel.clone()
            self._prev_imu_projected_gravity = projected_gravity.clone()
            self._prev_joint_pos = joint_pos.clone()
            self._prev_joint_vel = joint_vel.clone()

    def reset(self):
        """Reset all environments."""
        obs_dict, _ = self._env.reset()
        
        # Reset gait scheduler for all envs (uses config default for randomization)
        self.gait_scheduler.reset()
        
        # Reset phase oscillator
        self.gait_phase.zero_()
        
        # Reset episode tracking
        self.episode_rewards_sum.zero_()
        self.episode_steps.zero_()
        
        # Get current state and initialize previous state buffers
        ang_vel, projected_gravity, joint_pos, joint_vel = self._get_robot_state()
        self._init_prev_state(ang_vel, projected_gravity, joint_pos, joint_vel)
        
        self._last_obs = self._convert_obs(obs_dict)
        return self._last_obs

    def step(self, actions):
        """Step the environment."""
        # Store current state as previous before stepping
        ang_vel, projected_gravity, joint_pos, joint_vel = self._get_robot_state()
        
        clip_val = self.sensor_cfg.action_clip_value
        actions = torch.clamp(actions, -clip_val, clip_val)
        
        if torch.isnan(actions).any():
            actions = torch.nan_to_num(actions, nan=0.0)
        
        obs_dict, rewards, terminated, truncated, extras = self._env.step(actions)
        
        # Update gait scheduler with current velocity command
        cmd_manager = self._env.command_manager
        if hasattr(cmd_manager, 'get_command'):
            velocity_cmd = cmd_manager.get_command("base_velocity")
            if velocity_cmd is not None and velocity_cmd.shape[-1] >= 1:
                forward_vel = velocity_cmd[:, 0]
                self.gait_scheduler.set_commanded_velocity(forward_vel)
        
        # Advance gait scheduler
        self.gait_scheduler.step(self.sim_dt)
        
        # Advance phase oscillator synced to gait cycle frequency
        cycle_durations = self.gait_scheduler.cycle_durations
        phase_increment = (self.sim_dt / cycle_durations) * 2.0 * 3.14159265359
        self.gait_phase = (self.gait_phase + phase_increment) % (2.0 * 3.14159265359)
        
        # Combine terminated and truncated for RSL-RL
        dones = terminated | truncated
        
        # Track episode performance
        self.episode_rewards_sum += rewards.squeeze() if rewards.dim() > 1 else rewards
        self.episode_steps += 1
        
        # Handle completed episodes
        if dones.any():
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_indices.dim() == 0:
                done_indices = done_indices.unsqueeze(0)
            
            for idx in done_indices:
                idx_item = idx.item()
                ep_reward = self.episode_rewards_sum[idx_item].item()
                ep_length = self.episode_steps[idx_item].item()
                self._completed_episodes.append((ep_reward, ep_length))
            
            # Reset gait scheduler for done envs (uses config default for randomization)
            self.gait_scheduler.reset(done_indices)
            
            # Reset phase oscillator for done envs
            self.gait_phase[done_indices] = 0.0
            
            self.episode_rewards_sum[dones] = 0.0
            self.episode_steps[dones] = 0.0
            
            # Reset previous state for done envs
            new_ang_vel, new_proj_grav, new_joint_pos, new_joint_vel = self._get_robot_state()
            self._update_prev_state(new_ang_vel, new_proj_grav, new_joint_pos, new_joint_vel, done_indices)
        
        # Update previous state (for non-reset envs, this happened before step)
        non_done_mask = ~dones
        if non_done_mask.any():
            non_done_ids = non_done_mask.nonzero(as_tuple=False).squeeze(-1)
            if non_done_ids.dim() == 0:
                non_done_ids = non_done_ids.unsqueeze(0)
            self._prev_imu_ang_vel[non_done_ids] = ang_vel[non_done_ids]
            self._prev_imu_projected_gravity[non_done_ids] = projected_gravity[non_done_ids]
            self._prev_joint_pos[non_done_ids] = joint_pos[non_done_ids]
            self._prev_joint_vel[non_done_ids] = joint_vel[non_done_ids]
        
        # Convert observations
        self._last_obs = self._convert_obs(obs_dict)
        
        # RSL-RL expects time_outs in extras
        extras["time_outs"] = truncated
        extras["terrain_levels"] = self.terrain_levels.clone()
        
        # Ensure rewards are 1D and handle NaN
        if rewards.dim() > 1:
            rewards = rewards.squeeze(-1)
        
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            self._nan_reward_count += 1
            if self._nan_reward_count <= 5:
                print(f"WARNING: NaN/Inf in rewards (count: {self._nan_reward_count})")
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        
        return self._last_obs, rewards, dones, extras

    def get_observations(self):
        """Get current observations."""
        if self._last_obs is None:
            obs_dict = self._env.observation_manager.compute()
            self._last_obs = self._convert_obs(obs_dict)
        return self._last_obs

    def _convert_obs(self, obs_dict):
        """Convert Isaac Lab obs dict to TensorDict for RSL-RL."""
        policy_obs = obs_dict["policy"].float().contiguous()
        
        # Get gait scheduler outputs
        contact_states, foot_heights = self.gait_scheduler.get_gait_obs()
        
        ang_vel_scale = self.sensor_cfg.angular_velocity_scale
        joint_vel_scale = self.sensor_cfg.joint_velocity_scale
        
        # Get previous state (scaled consistently with current state)
        prev_ang_vel = self._prev_imu_ang_vel * ang_vel_scale
        prev_proj_grav = self._prev_imu_projected_gravity
        prev_joint_pos = self._prev_joint_pos
        prev_joint_vel = self._prev_joint_vel * joint_vel_scale
        
        # Phase oscillator observation (sin and cos for continuity)
        phase_obs = torch.stack([
            torch.sin(self.gait_phase),
            torch.cos(self.gait_phase)
        ], dim=-1)
        
        # Concatenate all observations
        policy_obs = torch.cat([
            policy_obs,
            prev_ang_vel,
            prev_proj_grav,
            prev_joint_pos,
            prev_joint_vel,
            contact_states,
            foot_heights,
            phase_obs,
        ], dim=-1)
        
        clip_val = self.sensor_cfg.obs_clip_value
        policy_obs = torch.clamp(policy_obs, -clip_val, clip_val)
        
        if torch.isnan(policy_obs).any() or torch.isinf(policy_obs).any():
            self._nan_obs_count += 1
            if self._nan_obs_count <= 5:
                print(f"WARNING: NaN/Inf in observations (count: {self._nan_obs_count})")
            policy_obs = torch.nan_to_num(policy_obs, nan=0.0, posinf=clip_val, neginf=-clip_val)
        
        return TensorDict({
            "policy": policy_obs
        }, batch_size=[self.num_envs])

    def get_gait_scheduler(self) -> GaitScheduler:
        """Get the gait scheduler for reward computation."""
        return self.gait_scheduler

    def close(self):
        """Clean up resources."""
        self._env.close()

    @property
    def unwrapped(self):
        """Return unwrapped environment."""
        return self._env
