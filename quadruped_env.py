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
        
        # Do initial reset to populate observations
        self.reset()

    def reset(self):
        """Reset all environments."""
        obs_dict, _ = self._env.reset()
        # Reset phase for all envs
        self.phase.zero_()
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
        
        # Reset phase for terminated envs
        if dones.any():
            self.phase[dones] = 0.0
        
        # Convert observations
        self._last_obs = self._convert_obs(obs_dict)
        
        # RSL-RL expects time_outs in extras
        extras["time_outs"] = truncated
        
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
