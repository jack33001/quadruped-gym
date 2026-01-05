"""
RSL-RL compatible wrapper for Isaac Lab quadruped environment.
"""
import math
import torch
from tensordict import TensorDict

from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.env import VecEnv

from gait_cfg import GaitScheduler, GaitSchedulerCfg
from sim_cfg import compute_projected_gravity


class IsaacLabVecEnvWrapper(VecEnv):
    """Wrapper to make Isaac Lab ManagerBasedRLEnv compatible with RSL-RL."""

    def __init__(self, env: ManagerBasedRLEnv, sensor_cfg=None):
        """Initialize wrapper."""
        self._env = env
        
        if sensor_cfg is None:
            from train_cfg import SensorCfg
            sensor_cfg = SensorCfg()
        self.sensor_cfg = sensor_cfg
        
        self._velocity_deadzone = sensor_cfg.velocity_deadzone
        
        self.num_envs = env.num_envs
        self.num_privileged_obs = None
        self.num_actions = env.action_manager.total_action_dim
        self.max_episode_length = int(env.max_episode_length)
        self.device = env.device

        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        gait_cfg = GaitSchedulerCfg()
        self.gait_scheduler = GaitScheduler(self.num_envs, self.device, gait_cfg)
        self.sim_dt = env.step_dt
        
        self._prev_imu_ang_vel = None
        self._prev_imu_projected_gravity = None
        self._prev_joint_pos = None
        self._prev_joint_vel = None
        self._prev_joint_torques = None
        self._prev_foot_contacts = None
        
        self._last_obs = None
        
        # Base: commands(3) + projected_gravity(3) + ang_vel(3) + joint_pos(8) + joint_vel(8) + last_action(8) = 33
        # New base observations: base_height(1) + foot_contact(4) + joint_torques(8) = 13
        # Prev state: prev_ang_vel(3) + prev_projected_gravity(3) + prev_joint_pos(8) + prev_joint_vel(8) = 22
        # New prev state: prev_joint_torques(8) + prev_foot_contacts(4) = 12
        # Gait: contact_states(4) + foot_xy_positions(8) + foot_heights(4) = 16
        # Per-leg phase: sin(phase)*4 + cos(phase)*4 = 8
        # Total additional: 22 + 12 + 16 + 8 = 58
        base_obs_dim = env.observation_manager.group_obs_dim["policy"][0]
        self.num_obs = base_obs_dim + 22 + 12 + 16 + 8
        
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
        self._do_initial_reset()

    def _do_initial_reset(self):
        """Perform initial reset to populate observation buffers."""
        obs_dict, _ = self._env.reset()
        
        self._apply_velocity_deadzone()
        
        self.gait_scheduler.reset()
        
        ang_vel, projected_gravity, joint_pos, joint_vel, joint_torques, foot_contacts = self._get_robot_state()
        self._init_prev_state(ang_vel, projected_gravity, joint_pos, joint_vel, joint_torques, foot_contacts)
        
        self._last_obs = self._convert_obs(obs_dict)

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
        
        projected_gravity = compute_projected_gravity(imu_sensor.data.quat_w)
        
        # Get joint state
        robot = self._env.scene["robot"]
        joint_pos = (robot.data.joint_pos - robot.data.default_joint_pos).clone()
        joint_vel = robot.data.joint_vel.clone()
        joint_torques = robot.data.applied_torque.clone()
        
        # Get foot contact state
        foot_contact_sensor = self._env.scene.sensors["foot_contact"]
        contact_forces = foot_contact_sensor.data.net_forces_w
        contact_threshold = self.sensor_cfg.foot_contact_threshold
        foot_contacts = (torch.norm(contact_forces, dim=-1) > contact_threshold).float()
        
        return ang_vel, projected_gravity, joint_pos, joint_vel, joint_torques, foot_contacts

    def _init_prev_state(self, ang_vel, projected_gravity, joint_pos, joint_vel, joint_torques, foot_contacts):
        """Initialize previous state buffers with current state."""
        self._prev_imu_ang_vel = ang_vel.clone()
        self._prev_imu_projected_gravity = projected_gravity.clone()
        self._prev_joint_pos = joint_pos.clone()
        self._prev_joint_vel = joint_vel.clone()
        self._prev_joint_torques = joint_torques.clone()
        self._prev_foot_contacts = foot_contacts.clone()

    def _update_prev_state(self, ang_vel, projected_gravity, joint_pos, joint_vel, joint_torques, foot_contacts, reset_ids=None):
        """Update previous state buffers."""
        if reset_ids is not None and len(reset_ids) > 0:
            self._prev_imu_ang_vel[reset_ids] = ang_vel[reset_ids]
            self._prev_imu_projected_gravity[reset_ids] = projected_gravity[reset_ids]
            self._prev_joint_pos[reset_ids] = joint_pos[reset_ids]
            self._prev_joint_vel[reset_ids] = joint_vel[reset_ids]
            self._prev_joint_torques[reset_ids] = joint_torques[reset_ids]
            self._prev_foot_contacts[reset_ids] = foot_contacts[reset_ids]
        else:
            self._prev_imu_ang_vel = ang_vel.clone()
            self._prev_imu_projected_gravity = projected_gravity.clone()
            self._prev_joint_pos = joint_pos.clone()
            self._prev_joint_vel = joint_vel.clone()
            self._prev_joint_torques = joint_torques.clone()
            self._prev_foot_contacts = foot_contacts.clone()

    def _apply_velocity_deadzone(self):
        """Apply deadzone to velocity commands, pushing values away from zero."""
        cmd_manager = self._env.command_manager
        if not hasattr(cmd_manager, 'get_command'):
            return
        
        velocity_cmd = cmd_manager.get_command("base_velocity")
        if velocity_cmd is None:
            return
        
        deadzone = self._velocity_deadzone
        if deadzone <= 0:
            return
        
        lin_vel_x = velocity_cmd[:, 0]
        
        in_positive_deadzone = (lin_vel_x > 0) & (lin_vel_x < deadzone)
        in_negative_deadzone = (lin_vel_x < 0) & (lin_vel_x > -deadzone)
        
        velocity_cmd[:, 0] = torch.where(
            in_positive_deadzone,
            deadzone,
            velocity_cmd[:, 0]
        )
        velocity_cmd[:, 0] = torch.where(
            in_negative_deadzone,
            -deadzone,
            velocity_cmd[:, 0]
        )

    def reset(self):
        """Reset all environments."""
        obs_dict, _ = self._env.reset()
        
        self._apply_velocity_deadzone()
        
        self.gait_scheduler.reset()
        
        self.episode_rewards_sum.zero_()
        self.episode_steps.zero_()
        
        self.episode_length_buf.zero_()
        
        ang_vel, projected_gravity, joint_pos, joint_vel, joint_torques, foot_contacts = self._get_robot_state()
        self._init_prev_state(ang_vel, projected_gravity, joint_pos, joint_vel, joint_torques, foot_contacts)
        
        self._last_obs = self._convert_obs(obs_dict)
        return self._last_obs

    def step(self, actions):
        """Step the environment."""
        ang_vel, projected_gravity, joint_pos, joint_vel, joint_torques, foot_contacts = self._get_robot_state()
        
        clip_val = self.sensor_cfg.action_clip_value
        actions = torch.clamp(actions, -clip_val, clip_val)
        
        if torch.isnan(actions).any():
            actions = torch.nan_to_num(actions, nan=0.0)
        
        obs_dict, rewards, terminated, truncated, extras = self._env.step(actions)
        
        self._apply_velocity_deadzone()
        
        cmd_manager = self._env.command_manager
        robot = self._env.scene["robot"]
        
        if hasattr(cmd_manager, 'get_command'):
            velocity_cmd = cmd_manager.get_command("base_velocity")
            if velocity_cmd is not None:
                self.gait_scheduler.set_commanded_velocity(velocity_cmd)
        
        current_vel = robot.data.root_lin_vel_w[:, :2]
        self.gait_scheduler.set_current_velocity(current_vel)
        
        self.gait_scheduler.step(self.sim_dt)
        
        self.episode_length_buf += 1
        
        time_out = self.episode_length_buf >= self.max_episode_length
        
        dones = terminated | truncated | time_out
        
        self.episode_rewards_sum += rewards.squeeze() if rewards.dim() > 1 else rewards
        self.episode_steps += 1
        
        if dones.any():
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_indices.dim() == 0:
                done_indices = done_indices.unsqueeze(0)
            
            for idx in done_indices:
                idx_item = idx.item()
                ep_reward = self.episode_rewards_sum[idx_item].item()
                ep_length = self.episode_steps[idx_item].item()
                self._completed_episodes.append((ep_reward, ep_length))
            
            self.gait_scheduler.reset(done_indices)
            
            self.episode_length_buf[done_indices] = 0
            
            self.episode_rewards_sum[dones] = 0.0
            self.episode_steps[dones] = 0.0
            
            new_ang_vel, new_proj_grav, new_joint_pos, new_joint_vel, new_joint_torques, new_foot_contacts = self._get_robot_state()
            self._update_prev_state(new_ang_vel, new_proj_grav, new_joint_pos, new_joint_vel, new_joint_torques, new_foot_contacts, done_indices)
        
        non_done_mask = ~dones
        if non_done_mask.any():
            non_done_ids = non_done_mask.nonzero(as_tuple=False).squeeze(-1)
            if non_done_ids.dim() == 0:
                non_done_ids = non_done_ids.unsqueeze(0)
            self._prev_imu_ang_vel[non_done_ids] = ang_vel[non_done_ids]
            self._prev_imu_projected_gravity[non_done_ids] = projected_gravity[non_done_ids]
            self._prev_joint_pos[non_done_ids] = joint_pos[non_done_ids]
            self._prev_joint_vel[non_done_ids] = joint_vel[non_done_ids]
            self._prev_joint_torques[non_done_ids] = joint_torques[non_done_ids]
            self._prev_foot_contacts[non_done_ids] = foot_contacts[non_done_ids]
        
        self._last_obs = self._convert_obs(obs_dict)
        
        extras["time_outs"] = time_out
        extras["terrain_levels"] = self.terrain_levels.clone()
        
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
        
        contact_states, foot_xy_positions, foot_heights = self.gait_scheduler.get_gait_obs()
        
        ang_vel_scale = self.sensor_cfg.angular_velocity_scale
        joint_vel_scale = self.sensor_cfg.joint_velocity_scale
        
        prev_ang_vel = self._prev_imu_ang_vel * ang_vel_scale
        prev_proj_grav = self._prev_imu_projected_gravity
        prev_joint_pos = self._prev_joint_pos
        prev_joint_vel = self._prev_joint_vel * joint_vel_scale
        prev_joint_torques = self._prev_joint_torques
        prev_foot_contacts = self._prev_foot_contacts
        
        leg_phases = self.gait_scheduler.get_leg_phases()
        phase_angles = leg_phases * 2.0 * 3.14159265359
        phase_sin = torch.sin(phase_angles)
        phase_cos = torch.cos(phase_angles)
        phase_obs = torch.cat([phase_sin, phase_cos], dim=-1)
        
        foot_xy_flat = foot_xy_positions.reshape(self.num_envs, -1)
        
        policy_obs = torch.cat([
            policy_obs,
            prev_ang_vel,
            prev_proj_grav,
            prev_joint_pos,
            prev_joint_vel,
            prev_joint_torques,
            prev_foot_contacts,
            contact_states,
            foot_xy_flat,
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

    def print_episode_length_stats(self):
        """Print statistics about episode_length_buf for debugging."""
        buf = self.episode_length_buf
        print(f"\nEpisode Length Buffer Stats:")
        print(f"  Min: {buf.min().item()}")
        print(f"  Max: {buf.max().item()}")
        print(f"  Mean: {buf.float().mean().item():.1f}")
        print(f"  Std: {buf.float().std().item():.1f}")
        print(f"  Unique values: {len(buf.unique())}")
        print(f"  Max episode length: {self.max_episode_length}")
