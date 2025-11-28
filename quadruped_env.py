"""
RSL-RL compatible wrapper for Isaac Lab quadruped environment.
"""
import math
import torch
from tensordict import TensorDict

from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.env import VecEnv


class IsaacLabVecEnvWrapper(VecEnv):
    """
    Wrapper to make Isaac Lab ManagerBasedRLEnv compatible with RSL-RL.
    """

    def __init__(self, env: ManagerBasedRLEnv):
        """Initialize wrapper."""
        self._env = env
        
        # VecEnv required attributes
        self.num_envs = env.num_envs
        self.num_obs = env.observation_manager.group_obs_dim["policy"][0]
        self.num_privileged_obs = None
        self.num_actions = env.action_manager.total_action_dim
        self.max_episode_length = int(env.max_episode_length)
        self.device = env.device

        # Buffers
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Phase tracking for gait clock signal
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase_freq = 2.0  # Hz - gait frequency
        self.phase_dt = env.step_dt * self.phase_freq * 2.0 * math.pi
        
        # Store last observation for get_observations()
        self._last_obs = None
        
        # Debug counters
        self._nan_obs_count = 0
        self._nan_reward_count = 0
        
        # Terrain curriculum tracking
        self._setup_terrain_curriculum()
        
        # Do initial reset to populate observations
        self.reset()

    def _setup_terrain_curriculum(self):
        """Setup terrain curriculum tracking buffers."""
        # Per-environment terrain level tracking
        self.terrain_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.terrain_types = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Episode performance tracking for curriculum
        self.episode_rewards_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_steps = torch.zeros(self.num_envs, device=self.device)
        
        # Velocity tracking for curriculum
        self.episode_vel_error_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_vel_cmd_sum = torch.zeros(self.num_envs, device=self.device)
        
        # Get terrain info if available
        self.num_terrain_rows = 5  # Default
        self.num_terrain_cols = 5  # Default
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

    def _get_velocity_tracking_error(self):
        """Compute current velocity tracking error."""
        try:
            # Get commanded velocity
            cmd_vel = self._env.command_manager.get_command("base_velocity")
            cmd_vel_x = cmd_vel[:, 0]  # Forward velocity command
            
            # Get actual velocity
            robot = self._env.scene["robot"]
            actual_vel = robot.data.root_lin_vel_b[:, 0]  # Forward velocity in body frame
            
            # Compute absolute error
            vel_error = torch.abs(actual_vel - cmd_vel_x)
            cmd_magnitude = torch.abs(cmd_vel_x).clamp(min=0.1)  # Avoid division by zero
            
            return vel_error, cmd_magnitude
        except Exception:
            # Fallback if velocity data unavailable
            return torch.zeros(self.num_envs, device=self.device), torch.ones(self.num_envs, device=self.device)

    def update_terrain_curriculum(self, survival_threshold: float = 0.7, velocity_threshold: float = 0.2):
        """
        Update terrain levels based on episode performance.
        
        Args:
            survival_threshold: Fraction of max episode length to survive (0-1).
            velocity_threshold: Max relative velocity error to pass (0-1, e.g., 0.2 = within 20%).
        """
        # Compute survival metric
        max_steps = self.max_episode_length
        survival_rate = self.episode_steps / max_steps
        
        # Compute velocity tracking metric (relative error)
        # Avoid division by zero with clamp
        vel_error_rate = self.episode_vel_error_sum / self.episode_vel_cmd_sum.clamp(min=1.0)
        
        # Find environments that just finished episodes (have accumulated steps)
        finished = self.episode_steps > 10  # Need at least some steps
        
        if not finished.any():
            return
        
        # Promote: survived long enough AND tracked velocity well
        promote_mask = finished & (survival_rate >= survival_threshold) & (vel_error_rate <= velocity_threshold)
        
        # Demote: poor survival OR very poor velocity tracking
        demote_mask = finished & ((survival_rate < survival_threshold * 0.5) | (vel_error_rate > velocity_threshold * 2.0))
        
        # Update terrain levels
        self.terrain_levels[promote_mask] = torch.clamp(
            self.terrain_levels[promote_mask] + 1, 
            max=self.max_terrain_level
        )
        self.terrain_levels[demote_mask] = torch.clamp(
            self.terrain_levels[demote_mask] - 1, 
            min=0
        )

    def reset(self):
        """Reset all environments."""
        obs_dict, _ = self._env.reset()
        # Reset phase for all envs
        self.phase.zero_()
        # Reset episode tracking
        self.episode_rewards_sum.zero_()
        self.episode_steps.zero_()
        self._last_obs = self._convert_obs(obs_dict)
        return self._last_obs

    def step(self, actions):
        """Step the environment."""
        # Clip actions for safety
        actions = torch.clamp(actions, -5.0, 5.0)
        
        # Check for NaN in actions
        if torch.isnan(actions).any():
            actions = torch.nan_to_num(actions, nan=0.0)
        
        obs_dict, rewards, terminated, truncated, extras = self._env.step(actions)
        
        # Update phase (wraps around at 2*pi)
        self.phase = (self.phase + self.phase_dt) % (2.0 * math.pi)
        
        # Combine terminated and truncated for RSL-RL
        dones = terminated | truncated
        
        # Track episode performance for terrain curriculum
        self.episode_rewards_sum += rewards.squeeze() if rewards.dim() > 1 else rewards
        self.episode_steps += 1
        
        # Track velocity error for curriculum
        vel_error, cmd_magnitude = self._get_velocity_tracking_error()
        self.episode_vel_error_sum += vel_error
        self.episode_vel_cmd_sum += cmd_magnitude
        
        # Reset phase and episode tracking for terminated envs
        if dones.any():
            self.phase[dones] = 0.0
            # Reset episode tracking for done envs (after curriculum update uses them)
            self.episode_rewards_sum[dones] = 0.0
            self.episode_steps[dones] = 0.0
            self.episode_vel_error_sum[dones] = 0.0
            self.episode_vel_cmd_sum[dones] = 0.0
        
        # Convert observations
        self._last_obs = self._convert_obs(obs_dict)
        
        # RSL-RL expects time_outs in extras
        extras["time_outs"] = truncated
        
        # Add terrain curriculum info to extras
        extras["terrain_levels"] = self.terrain_levels.clone()
        
        # Ensure rewards are 1D and handle NaN
        if rewards.dim() > 1:
            rewards = rewards.squeeze(-1)
        
        # Check for NaN in rewards
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
        # Ensure observations are float32 and contiguous
        policy_obs = obs_dict["policy"].float().contiguous()
        
        # Append phase signal (sin and cos for continuity)
        phase_obs = torch.stack([
            torch.sin(self.phase),
            torch.cos(self.phase)
        ], dim=-1)
        
        # Concatenate phase to policy observations
        policy_obs = torch.cat([policy_obs, phase_obs], dim=-1)
        
        # Clip observations to reasonable range
        policy_obs = torch.clamp(policy_obs, -100.0, 100.0)
        
        # Check for NaN/Inf
        if torch.isnan(policy_obs).any() or torch.isinf(policy_obs).any():
            self._nan_obs_count += 1
            if self._nan_obs_count <= 5:
                print(f"WARNING: NaN/Inf in observations (count: {self._nan_obs_count})")
            policy_obs = torch.nan_to_num(policy_obs, nan=0.0, posinf=100.0, neginf=-100.0)
        
        return TensorDict({
            "policy": policy_obs
        }, batch_size=[self.num_envs])

    def close(self):
        """Clean up resources."""
        self._env.close()

    @property
    def unwrapped(self):
        """Return unwrapped environment."""
        return self._env
