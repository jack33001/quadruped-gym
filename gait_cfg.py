"""
Gait configuration and scheduler for quadruped locomotion.

Defines gait patterns (trot, bound, pronk, etc.) with per-leg phase offsets,
stance/swing durations, and provides a deterministic gait scheduler with
Raibert heuristic foot placement.
"""
import torch
from dataclasses import dataclass
from enum import IntEnum


class GaitType(IntEnum):
    """Gait type enumeration."""
    TROT = 0
    RUN = 1
    BOUND = 2
    PRONK = 3
    AMBLE = 4
    HOP = 5


@dataclass
class GaitParams:
    """Parameters for a single gait pattern.
    
    Phase offsets are fractions of the gait cycle [0, 1).
    Leg order: [front_right, front_left, rear_right, rear_left]
    
    Target step frequency at 1 m/s is 7.5 Hz, giving cycle_duration ~0.133s.
    Duty factor (stance fraction) varies by gait type.
    """
    name: str
    stance_duration: float
    swing_duration: float
    phase_offsets: tuple
    
    @property
    def cycle_duration(self) -> float:
        return self.stance_duration + self.swing_duration
    
    @property
    def duty_factor(self) -> float:
        """Fraction of cycle spent in stance."""
        return self.stance_duration / self.cycle_duration


GAIT_PARAMS = {
    GaitType.TROT: GaitParams(
        name="trot",
        stance_duration=0.08,
        swing_duration=0.053,
        phase_offsets=(0.0, 0.5, 0.5, 0.0),
    ),
    GaitType.RUN: GaitParams(
        name="run",
        stance_duration=0.066,
        swing_duration=0.067,
        phase_offsets=(0.0, 0.5, 0.25, 0.75),
    ),
    GaitType.BOUND: GaitParams(
        name="bound",
        stance_duration=0.073,
        swing_duration=0.06,
        phase_offsets=(0.0, 0.0, 0.5, 0.5),
    ),
    GaitType.PRONK: GaitParams(
        name="pronk",
        stance_duration=0.073,
        swing_duration=0.06,
        phase_offsets=(0.0, 0.0, 0.0, 0.0),
    ),
    GaitType.AMBLE: GaitParams(
        name="amble",
        stance_duration=0.08,
        swing_duration=0.053,
        phase_offsets=(0.0, 0.5, 0.25, 0.75),
    ),
    GaitType.HOP: GaitParams(
        name="hop",
        stance_duration=0.053,
        swing_duration=0.08,
        phase_offsets=(0.0, 0.0, 0.0, 0.0),
    ),
}


@dataclass
class GaitSchedulerCfg:
    """Configuration for the gait scheduler."""
    num_legs: int = 4
    swing_height: float = 0.05
    default_gait: GaitType = GaitType.AMBLE
    randomize_gait: bool = True
    
    enabled_gaits: tuple = (GaitType.AMBLE, GaitType.TROT, GaitType.BOUND)
    
    base_velocity: float = 1.0
    min_cycle_scale: float = 0.75
    max_cycle_scale: float = 1.25
    
    allow_gait_switching: bool = True
    min_gait_switch_interval: float = 1.5
    gait_switch_probability: float = 0.002
    
    # Raibert heuristic parameters
    raibert_gain: float = 0.3
    hip_offset_x: float = 0.1
    hip_offset_y: float = 0.08
    
    # Leg order: [front_right, front_left, rear_right, rear_left]
    # Signs: (x_sign, y_sign) where positive x is forward, positive y is left
    leg_signs: tuple = ((1, -1), (1, 1), (-1, -1), (-1, 1))


class GaitScheduler:
    """Deterministic gait scheduler with Raibert heuristic foot placement.
    
    Manages per-leg phases and outputs desired contact states and foot positions
    based on the current gait type, phase progression, and velocity commands.
    """
    
    def __init__(self, num_envs: int, device: torch.device, cfg: GaitSchedulerCfg = None):
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg if cfg is not None else GaitSchedulerCfg()
        
        self.enabled_gaits = torch.tensor(self.cfg.enabled_gaits, device=device, dtype=torch.long)
        self.num_enabled_gaits = len(self.enabled_gaits)
        
        self.leg_phases = torch.zeros(num_envs, self.cfg.num_legs, device=device)
        self.gait_types = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        self.commanded_velocity = torch.zeros(num_envs, 3, device=device)
        self.commanded_velocity[:, 0] = self.cfg.base_velocity
        
        self.current_velocity = torch.zeros(num_envs, 3, device=device)
        
        self.time_since_gait_switch = torch.zeros(num_envs, device=device)
        
        self._setup_hip_positions()
        self._cache_gait_params()
    
    def _setup_hip_positions(self):
        """Setup nominal hip positions relative to body center."""
        leg_signs = torch.tensor(self.cfg.leg_signs, device=self.device, dtype=torch.float)
        
        self.hip_positions = torch.zeros(self.cfg.num_legs, 3, device=self.device)
        self.hip_positions[:, 0] = leg_signs[:, 0] * self.cfg.hip_offset_x
        self.hip_positions[:, 1] = leg_signs[:, 1] * self.cfg.hip_offset_y
        self.hip_positions[:, 2] = 0.0
    
    def _cache_gait_params(self):
        """Pre-compute gait parameters as tensors for fast lookup."""
        num_gaits = len(GaitType)
        
        self.base_cycle_durations = torch.zeros(num_gaits, device=self.device)
        self.duty_factors = torch.zeros(num_gaits, device=self.device)
        self.phase_offsets = torch.zeros(num_gaits, self.cfg.num_legs, device=self.device)
        
        for gait_type in GaitType:
            params = GAIT_PARAMS[gait_type]
            self.base_cycle_durations[gait_type] = params.cycle_duration
            self.duty_factors[gait_type] = params.duty_factor
            self.phase_offsets[gait_type] = torch.tensor(
                params.phase_offsets, device=self.device
            )
    
    @property
    def cycle_durations(self) -> torch.Tensor:
        """Get velocity-scaled cycle durations for each environment."""
        base_durations = self.base_cycle_durations[self.gait_types]
        
        velocity_magnitude = torch.norm(self.commanded_velocity[:, :2], dim=-1)
        
        scale = self.cfg.base_velocity / velocity_magnitude.clamp(min=0.1)
        scale = scale.clamp(self.cfg.min_cycle_scale, self.cfg.max_cycle_scale)
        
        return base_durations * scale
    
    def set_commanded_velocity(self, velocity: torch.Tensor):
        """Update the commanded velocity for trajectory computation.
        
        Args:
            velocity: Velocity command tensor. Can be:
                - (num_envs,) for forward velocity only
                - (num_envs, 2) for x, y velocity
                - (num_envs, 3) for x, y, yaw velocity
        """
        if velocity.dim() == 1:
            self.commanded_velocity[:, 0] = velocity
            self.commanded_velocity[:, 1] = 0.0
            self.commanded_velocity[:, 2] = 0.0
        elif velocity.shape[-1] == 2:
            self.commanded_velocity[:, :2] = velocity
            self.commanded_velocity[:, 2] = 0.0
        else:
            self.commanded_velocity = velocity.clone()
    
    def set_current_velocity(self, velocity: torch.Tensor):
        """Update the current body velocity for Raibert heuristic.
        
        Args:
            velocity: Current velocity tensor (num_envs, 3) for x, y, yaw.
        """
        if velocity.dim() == 1:
            self.current_velocity[:, 0] = velocity
        elif velocity.shape[-1] >= 2:
            self.current_velocity[:, :velocity.shape[-1]] = velocity[:, :3] if velocity.shape[-1] > 3 else velocity
    
    def reset(self, env_ids: torch.Tensor = None, randomize_gait: bool = None):
        """Reset phases for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        if len(env_ids) == 0:
            return
        
        self.leg_phases[env_ids] = 0.0
        self.time_since_gait_switch[env_ids] = 0.0
        self.current_velocity[env_ids] = 0.0
        
        should_randomize = randomize_gait if randomize_gait is not None else self.cfg.randomize_gait
        
        if should_randomize:
            random_indices = torch.randint(
                0, self.num_enabled_gaits, (len(env_ids),), device=self.device
            )
            self.gait_types[env_ids] = self.enabled_gaits[random_indices]
        else:
            self.gait_types[env_ids] = self.cfg.default_gait
    
    def step(self, dt: float):
        """Advance phases by dt seconds."""
        cycle_dur = self.cycle_durations
        phase_increment = dt / cycle_dur.unsqueeze(-1)
        self.leg_phases = (self.leg_phases + phase_increment) % 1.0
        
        self.time_since_gait_switch += dt
        
        if self.cfg.allow_gait_switching:
            self._maybe_switch_gaits()
    
    def _maybe_switch_gaits(self):
        """Randomly switch gaits for environments that meet the criteria."""
        can_switch = self.time_since_gait_switch >= self.cfg.min_gait_switch_interval
        
        switch_roll = torch.rand(self.num_envs, device=self.device)
        should_switch = can_switch & (switch_roll < self.cfg.gait_switch_probability)
        
        if should_switch.any():
            switch_indices = should_switch.nonzero(as_tuple=False).squeeze(-1)
            if switch_indices.dim() == 0:
                switch_indices = switch_indices.unsqueeze(0)
            
            random_indices = torch.randint(
                0, self.num_enabled_gaits, (len(switch_indices),), device=self.device
            )
            new_gaits = self.enabled_gaits[random_indices]
            self.gait_types[switch_indices] = new_gaits
            
            self.time_since_gait_switch[switch_indices] = 0.0
    
    def get_contact_states(self) -> torch.Tensor:
        """Get desired contact state for each leg.
        
        Returns:
            Tensor of shape (num_envs, num_legs) with 1.0 for stance, 0.0 for swing.
        """
        offsets = self.phase_offsets[self.gait_types]
        effective_phase = (self.leg_phases + offsets) % 1.0
        duty = self.duty_factors[self.gait_types].unsqueeze(-1)
        contact_states = (effective_phase < duty).float()
        return contact_states
    
    def _compute_raibert_foothold(self) -> torch.Tensor:
        """Compute desired foot positions using Raibert heuristic.
        
        The Raibert heuristic places feet at:
            p_foot = p_hip + v_current * T_stance/2 + K * (v_current - v_desired)
        
        Returns:
            Tensor of shape (num_envs, num_legs, 3) with desired foot positions
            in body frame.
        """
        duty = self.duty_factors[self.gait_types]
        cycle_dur = self.cycle_durations
        stance_duration = duty * cycle_dur
        
        v_current = self.current_velocity[:, :2]
        v_desired = self.commanded_velocity[:, :2]
        
        velocity_offset = v_current * (stance_duration.unsqueeze(-1) / 2.0)
        
        velocity_error = v_current - v_desired
        feedback_offset = self.cfg.raibert_gain * velocity_error
        
        total_offset = velocity_offset + feedback_offset
        
        foot_positions = self.hip_positions.unsqueeze(0).expand(self.num_envs, -1, -1).clone()
        
        foot_positions[:, :, 0] += total_offset[:, 0:1]
        foot_positions[:, :, 1] += total_offset[:, 1:2]
        
        return foot_positions
    
    def get_foot_positions(self) -> torch.Tensor:
        """Get desired foot positions for each leg.
        
        During stance: foot at Raibert touchdown position, z = 0
        During swing: foot follows trajectory from liftoff to next touchdown
        
        Returns:
            Tensor of shape (num_envs, num_legs, 3) with desired positions.
        """
        offsets = self.phase_offsets[self.gait_types]
        effective_phase = (self.leg_phases + offsets) % 1.0
        duty = self.duty_factors[self.gait_types].unsqueeze(-1)
        
        in_swing = effective_phase >= duty
        
        swing_duration_frac = 1.0 - duty
        swing_progress = torch.where(
            in_swing,
            (effective_phase - duty) / swing_duration_frac.clamp(min=1e-6),
            torch.zeros_like(effective_phase)
        )
        swing_progress = swing_progress.clamp(0.0, 1.0)
        
        touchdown_positions = self._compute_raibert_foothold()
        
        liftoff_positions = self.hip_positions.unsqueeze(0).expand(self.num_envs, -1, -1).clone()
        
        interp = swing_progress.unsqueeze(-1)
        xy_positions = liftoff_positions * (1.0 - interp) + touchdown_positions * interp
        
        z_height = 4.0 * self.cfg.swing_height * swing_progress * (1.0 - swing_progress)
        
        foot_positions = xy_positions.clone()
        foot_positions[:, :, 2] = torch.where(
            in_swing,
            z_height,
            torch.zeros_like(z_height)
        )
        
        return foot_positions
    
    def get_foot_heights(self) -> torch.Tensor:
        """Get desired foot height for each leg (z component only).
        
        Returns:
            Tensor of shape (num_envs, num_legs) with desired heights.
        """
        return self.get_foot_positions()[:, :, 2]
    
    def get_foot_xy_positions(self) -> torch.Tensor:
        """Get desired foot x, y positions for each leg.
        
        Returns:
            Tensor of shape (num_envs, num_legs, 2) with desired x, y positions.
        """
        return self.get_foot_positions()[:, :, :2]
    
    def get_leg_phases(self) -> torch.Tensor:
        """Get the phase [0, 1) for each leg, including phase offsets.
        
        Returns:
            Tensor of shape (num_envs, num_legs) with phase values in [0, 1).
        """
        offsets = self.phase_offsets[self.gait_types]
        effective_phase = (self.leg_phases + offsets) % 1.0
        return effective_phase
    
    def get_normalized_phase(self) -> torch.Tensor:
        """Get the normalized gait phase [0, 1) for each environment.
        
        Uses the first leg's phase as the reference phase for the gait cycle.
        
        Returns:
            Tensor of shape (num_envs,) with phase values in [0, 1).
        """
        return self.leg_phases[:, 0]
    
    def get_gait_obs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get gait scheduler outputs for observation.
        
        Returns:
            Tuple of (contact_states, foot_xy_positions, foot_heights):
                - contact_states: (num_envs, num_legs)
                - foot_xy_positions: (num_envs, num_legs, 2)
                - foot_heights: (num_envs, num_legs)
        """
        contact_states = self.get_contact_states()
        foot_positions = self.get_foot_positions()
        return contact_states, foot_positions[:, :, :2], foot_positions[:, :, 2]
    
    def get_current_gait_names(self) -> list[str]:
        """Get human-readable gait names for each environment."""
        names = []
        for gait_idx in self.gait_types.cpu().numpy():
            gait_type = GaitType(gait_idx)
            names.append(GAIT_PARAMS[gait_type].name)
        return names
