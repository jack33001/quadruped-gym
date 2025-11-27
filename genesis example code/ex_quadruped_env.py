"""
RSL-RL compatible quadruped environment.
Clean separation of concerns with modular reward functions.
"""
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import numpy as np
import torch
from tensordict import TensorDict

import genesis as gs
from rsl_rl.env import VecEnv

from configs import EnvConfig, ObsConfig, RewardConfig, CommandConfig


class QuadrupedEnv(VecEnv):
    """
    Quadruped locomotion environment compatible with RSL-RL.
    
    Follows RSL-RL conventions:
    - Inherits from VecEnv
    - get_observations() returns TensorDict
    - step() returns (TensorDict, rewards, dones, extras)
    - Rewards are shape (num_envs,)
    """
    
    def __init__(
        self,
        num_envs: int = 128,
        env_cfg: EnvConfig = EnvConfig(),
        obs_cfg: ObsConfig = ObsConfig(),
        reward_cfg: RewardConfig = RewardConfig(),
        command_cfg: CommandConfig = CommandConfig(),
    ):
        """Initialize environment."""
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        
        # Update num_envs from parameter
        self.env_cfg.num_envs = num_envs
        
        # VecEnv required attributes
        self.num_envs = num_envs
        self.num_obs = obs_cfg.num_obs
        self.num_privileged_obs = None
        self.num_actions = env_cfg.num_actions
        self.max_episode_length = int(env_cfg.episode_length_s / env_cfg.dt)
        self.device = gs.device if hasattr(gs, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create scene (Genesis should already be initialized)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=env_cfg.dt,
                substeps=env_cfg.substeps,
            ),
            show_viewer=env_cfg.show_viewer,
        )
        
        # Add plane and robot
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=env_cfg.robot_file))
        
        # Build scene
        self.scene.build(
            n_envs=num_envs,
            env_spacing=(env_cfg.env_spacing, env_cfg.env_spacing)
        )
        
        # Get DOF indices
        self.dof_indices = [
            self.robot.get_joint(name).dof_idx_local 
            for name in env_cfg.joint_names
        ]
        
        # Get foot link indices
        self.foot_link_indices = [
            self.robot.get_link(name).idx_local 
            for name in env_cfg.foot_link_names
        ]
        
        # Set PD gains
        kp_array = np.full(env_cfg.num_actions, env_cfg.kp)
        kd_array = np.full(env_cfg.num_actions, env_cfg.kd)
        self.robot.set_dofs_kp(kp=kp_array, dofs_idx_local=self.dof_indices)
        self.robot.set_dofs_kv(kv=kd_array, dofs_idx_local=self.dof_indices)
        
        # Buffers
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.reset_buf = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        
        self.last_actions = torch.zeros(num_envs, env_cfg.num_actions, device=self.device)
        self.commands = torch.zeros(num_envs, command_cfg.num_commands, device=self.device)
        
        # Default joint positions
        self.default_dof_pos = torch.tensor(
            env_cfg.default_joint_angles, 
            device=self.device, 
            dtype=torch.float32
        )
        
        # Initialize
        self.reset()
    
    def reset(self):
        """Reset all environments. Returns TensorDict observations."""
        self._reset_envs(torch.arange(self.num_envs, device=self.device))
        return self.get_observations()
    
    def _reset_envs(self, env_ids):
        """Reset specified environments."""
        if len(env_ids) == 0:
            return
        
        # Reset to initial pose with small randomization
        init_pos = torch.tensor(self.env_cfg.base_init_pos, device=self.device)
        init_quat = torch.tensor(self.env_cfg.base_init_quat, device=self.device)
        
        reset_pos = init_pos.unsqueeze(0).repeat(len(env_ids), 1)
        reset_quat = init_quat.unsqueeze(0).repeat(len(env_ids), 1)
        
        # Randomize base height (z)
        height_min, height_max = self.env_cfg.base_height_range
        reset_pos[:, 2] = torch.rand(len(env_ids), device=self.device) * (height_max - height_min) + height_min
        
        # Reset joint positions with noise
        reset_dof_pos = self.default_dof_pos.unsqueeze(0).repeat(len(env_ids), 1)
        reset_dof_pos += torch.randn_like(reset_dof_pos) * 0.1
        
        # Apply resets
        self.robot.set_dofs_position(
            reset_dof_pos, 
            self.dof_indices, 
            zero_velocity=True, 
            envs_idx=env_ids.cpu().numpy()
        )
        self.robot.set_pos(
            reset_pos, 
            envs_idx=env_ids.cpu().numpy(), 
            zero_velocity=True
        )
        self.robot.set_quat(
            reset_quat, 
            envs_idx=env_ids.cpu().numpy(), 
            zero_velocity=True
        )
        
        # Set control targets to default
        self.robot.control_dofs_position(
            self.default_dof_pos.unsqueeze(0).repeat(len(env_ids), 1),
            self.dof_indices,
            envs_idx=env_ids.cpu().numpy()
        )
        
        # Settle simulation
        for _ in range(5):
            self.scene.step()
        
        # Reset buffers
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.last_actions[env_ids] = 0.0
        
        # Sample new commands
        self.commands[env_ids] = self._sample_commands(len(env_ids))
    
    def _sample_commands(self, num_commands: int):
        """Sample random velocity commands."""
        commands = torch.zeros(num_commands, 3, device=self.device)
        
        # Sample from configured ranges
        commands[:, 0] = torch.rand(num_commands, device=self.device) * (
            self.command_cfg.lin_vel_x_range[1] - self.command_cfg.lin_vel_x_range[0]
        ) + self.command_cfg.lin_vel_x_range[0]
        
        commands[:, 1] = torch.rand(num_commands, device=self.device) * (
            self.command_cfg.lin_vel_y_range[1] - self.command_cfg.lin_vel_y_range[0]
        ) + self.command_cfg.lin_vel_y_range[0]
        
        commands[:, 2] = torch.rand(num_commands, device=self.device) * (
            self.command_cfg.ang_vel_range[1] - self.command_cfg.ang_vel_range[0]
        ) + self.command_cfg.ang_vel_range[0]
        
        return commands
    
    def get_observations(self):
        """
        Get observations for all environments.
        Returns TensorDict with 'policy' key for RSL-RL.
        """
        # Get robot state
        dof_pos = self.robot.get_dofs_position(self.dof_indices)  # (num_envs, num_dofs)
        dof_vel = self.robot.get_dofs_velocity(self.dof_indices)
        base_quat = self.robot.get_quat()  # (num_envs, 4)
        base_ang_vel = self.robot.get_ang()  # (num_envs, 3)
        
        # Compute gravity in body frame
        gravity_body = self._compute_gravity_in_body_frame(base_quat)
        
        # Apply observation scaling
        gravity_scaled = gravity_body / 9.81
        ang_vel_scaled = base_ang_vel * self.obs_cfg.obs_scales['ang_vel']
        dof_pos_scaled = dof_pos * self.obs_cfg.obs_scales['dof_pos']
        dof_vel_scaled = dof_vel * self.obs_cfg.obs_scales['dof_vel']
        
        # Concatenate observations
        obs = torch.cat([
            gravity_scaled,           # 3
            self.commands,            # 3
            ang_vel_scaled,           # 3
            dof_pos_scaled,           # 8
            dof_vel_scaled,           # 8
            self.last_actions,        # 8
        ], dim=-1)  # Total: 33
        
        # Return as TensorDict for RSL-RL
        return TensorDict({
            'policy': obs
        }, batch_size=[self.num_envs])
    
    def step(self, actions):
        """
        Step the environment.
        
        Args:
            actions: torch.Tensor of shape (num_envs, num_actions)
        
        Returns:
            obs: TensorDict with observations
            rewards: torch.Tensor of shape (num_envs,)
            dones: torch.Tensor of shape (num_envs,)
            extras: dict with additional info
        """
        # Clip and scale actions
        actions = torch.clip(actions, -100.0, 100.0)
        targets = self.default_dof_pos + actions * self.env_cfg.action_scale
        
        # Send control commands
        self.robot.control_dofs_position(targets, self.dof_indices)
        
        # Step simulation
        self.scene.step()
        
        # Get observations
        obs_dict = self.get_observations()
        
        # Compute rewards
        rewards, reward_components = self._compute_rewards(actions)
        
        # Check terminations
        dones = self._check_termination()
        
        # Update episode lengths
        self.episode_length_buf += 1
        
        # Reset terminated environments
        reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_envs(reset_env_ids)
        
        # Update buffers
        self.last_actions[:] = actions
        
        # Extras for RSL-RL
        # Put reward components in "log" dict so RSL-RL automatically logs them
        extras = {
            'time_outs': dones,  # RSL-RL uses this for GAE
            'log': {
                f'Reward/{k}': v.mean().item() for k, v in reward_components.items()
            }
        }
        
        return obs_dict, rewards, dones, extras
    
    def _compute_rewards(self, actions):
        """Compute reward function."""
        # Get robot state
        base_lin_vel = self.robot.get_vel()  # (num_envs, 3)
        base_ang_vel = self.robot.get_ang()  # (num_envs, 3)
        base_pos = self.robot.get_pos()  # (num_envs, 3)
        dof_pos = self.robot.get_dofs_position(self.dof_indices)
        
        # Initialize reward
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Store individual components for logging (SCALED versions)
        reward_components = {}
        
        # Linear velocity tracking (xy plane only)
        lin_vel_xy_error = torch.sum(
            torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), 
            dim=1
        )
        rew_lin_vel_xy_tracking = torch.exp(-lin_vel_xy_error / self.reward_cfg.tracking_sigma)
        scaled_lin_vel_xy = self.reward_cfg.scales['tracking_lin_vel_xy'] * rew_lin_vel_xy_tracking
        reward_components['tracking_lin_vel_xy'] = scaled_lin_vel_xy
        rewards += scaled_lin_vel_xy
        
        # Angular velocity tracking (yaw only)
        ang_vel_yaw_error = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])
        rew_ang_vel_yaw_tracking = torch.exp(-ang_vel_yaw_error / self.reward_cfg.tracking_sigma)
        scaled_ang_vel_yaw = self.reward_cfg.scales['tracking_ang_vel_yaw'] * rew_ang_vel_yaw_tracking
        reward_components['tracking_ang_vel_yaw'] = scaled_ang_vel_yaw
        rewards += scaled_ang_vel_yaw
        
        # Penalize pitch and roll angular velocities (should always be near zero)
        rew_ang_vel_pitch_roll = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
        scaled_ang_vel_pitch_roll = self.reward_cfg.scales['ang_vel_pitch_roll'] * rew_ang_vel_pitch_roll
        reward_components['ang_vel_pitch_roll'] = scaled_ang_vel_pitch_roll
        rewards += scaled_ang_vel_pitch_roll
        
        # Penalize z-axis velocity (jumping/falling)
        rew_lin_vel_z = torch.square(base_lin_vel[:, 2])
        scaled_lin_vel_z = self.reward_cfg.scales['lin_vel_z'] * rew_lin_vel_z
        reward_components['lin_vel_z'] = scaled_lin_vel_z
        rewards += scaled_lin_vel_z
        
        # Penalize base height deviation
        rew_base_height = torch.square(base_pos[:, 2] - self.reward_cfg.base_height_target)
        scaled_base_height = self.reward_cfg.scales['base_height'] * rew_base_height
        reward_components['base_height'] = scaled_base_height
        rewards += scaled_base_height
        
        # Penalize action rate (smoothness)
        rew_action_rate = torch.sum(torch.square(self.last_actions - actions), dim=1)
        scaled_action_rate = self.reward_cfg.scales['action_rate'] * rew_action_rate
        reward_components['action_rate'] = scaled_action_rate
        rewards += scaled_action_rate
        
        # Penalize deviation from default pose
        rew_default_pose = torch.sum(torch.abs(dof_pos - self.default_dof_pos), dim=1)
        scaled_default_pose = self.reward_cfg.scales['similar_to_default'] * rew_default_pose
        reward_components['similar_to_default'] = scaled_default_pose
        rewards += scaled_default_pose
        
        return rewards, reward_components
    
    def _check_termination(self):
        """Check if environments should terminate."""
        # Get orientation
        base_quat = self.robot.get_quat()
        gravity_body = self._compute_gravity_in_body_frame(base_quat)
        
        # Check if robot fell over (roll/pitch too large)
        roll_pitch_threshold = np.cos(
            np.radians(self.env_cfg.termination_if_roll_greater_than)
        )
        upright = -gravity_body[:, 2] / 9.81
        fell = upright < roll_pitch_threshold
        
        # Check timeout
        timeout = self.episode_length_buf >= self.max_episode_length
        
        dones = fell | timeout
        
        # LOG TERMINATION REASONS (only in eval mode with single env)
        if self.num_envs == 1 and dones.any():
            env_id = 0
            if fell[env_id]:
                # Calculate actual roll/pitch for debugging
                base_pos = self.robot.get_pos()
                print(f"FELL: upright={upright[env_id].item():.3f} (threshold={roll_pitch_threshold:.3f}), "
                      f"height={base_pos[env_id, 2].item():.3f}m, step={self.episode_length_buf[env_id].item()}")
            elif timeout[env_id]:
                print(f"TIMEOUT: Completed full episode ({self.max_episode_length} steps)")
        
        return dones
    
    def _compute_gravity_in_body_frame(self, quat):
        """
        Project gravity vector into body frame.
        
        Args:
            quat: torch.Tensor of shape (num_envs, 4) in [w, x, y, z] format
        
        Returns:
            gravity_body: torch.Tensor of shape (num_envs, 3)
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Rotation matrix elements for transforming [0, 0, -g] to body frame
        g = -9.81
        g_x = 2 * (x*z - w*y) * g
        g_y = 2 * (y*z + w*x) * g
        g_z = (w*w - x*x - y*y + z*z) * g
        
        return torch.stack([g_x, g_y, g_z], dim=1)
    
    def close(self):
        """Clean up resources."""
        pass
