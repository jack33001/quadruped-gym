"""
Gait configuration and scheduler for quadruped locomotion.

Defines gait patterns (trot, bound, pronk, etc.) with per-leg phase offsets,
stance/swing durations, and provides a deterministic gait scheduler.
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
    
    base_velocity: float = 1.0
    min_cycle_scale: float = 0.5
    max_cycle_scale: float = 1.5
    
    allow_gait_switching: bool = True
    min_gait_switch_interval: float = 1.5
    gait_switch_probability: float = 0.002


class GaitScheduler:
    """Deterministic gait scheduler.
    
    Manages per-leg phases and outputs desired contact states and foot heights
    based on the current gait type and phase progression.
    
    Cycle duration scales with commanded velocity - faster commands result in
    shorter cycle times (faster stepping).
    """
    
    def __init__(self, num_envs: int, device: torch.device, cfg: GaitSchedulerCfg = None):
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg if cfg is not None else GaitSchedulerCfg()
        
        self.leg_phases = torch.zeros(num_envs, self.cfg.num_legs, device=device)
        self.gait_types = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        self.commanded_velocity = torch.ones(num_envs, device=device) * self.cfg.base_velocity
        
        self.time_since_gait_switch = torch.zeros(num_envs, device=device)
        
        self._cache_gait_params()
    
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
        
        velocity_magnitude = torch.abs(self.commanded_velocity)
        
        scale = self.cfg.base_velocity / velocity_magnitude.clamp(min=0.1)
        scale = scale.clamp(self.cfg.min_cycle_scale, self.cfg.max_cycle_scale)
        
        return base_durations * scale
    
    def set_commanded_velocity(self, velocity: torch.Tensor):
        """Update the commanded velocity for cycle duration scaling.
        
        Args:
            velocity: Forward velocity command (num_envs,) or scalar
        """
        if velocity.dim() == 0:
            self.commanded_velocity.fill_(velocity.item())
        else:
            self.commanded_velocity = velocity.clone()
    
    def reset(self, env_ids: torch.Tensor = None, randomize_gait: bool = None):
        """Reset phases for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        if len(env_ids) == 0:
            return
        
        self.leg_phases[env_ids] = 0.0
        self.time_since_gait_switch[env_ids] = 0.0
        
        should_randomize = randomize_gait if randomize_gait is not None else self.cfg.randomize_gait
        
        if should_randomize:
            num_gaits = len(GaitType)
            self.gait_types[env_ids] = torch.randint(
                0, num_gaits, (len(env_ids),), device=self.device
            )
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
            
            num_gaits = len(GaitType)
            new_gaits = torch.randint(0, num_gaits, (len(switch_indices),), device=self.device)
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
    
    def get_foot_heights(self) -> torch.Tensor:
        """Get desired foot height for each leg.
        
        During stance: height = 0
        During swing: height follows parabolic trajectory with apex at swing_height
        
        Returns:
            Tensor of shape (num_envs, num_legs) with desired heights.
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
        
        foot_heights = 4.0 * self.cfg.swing_height * swing_progress * (1.0 - swing_progress)
        foot_heights = torch.where(in_swing, foot_heights, torch.zeros_like(foot_heights))
        
        return foot_heights
    
    def get_gait_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get gait scheduler outputs for observation.
        
        Returns:
            Tuple of (contact_states, foot_heights), each (num_envs, num_legs).
        """
        return self.get_contact_states(), self.get_foot_heights()
    
    def get_current_gait_names(self) -> list[str]:
        """Get human-readable gait names for each environment."""
        names = []
        for gait_idx in self.gait_types.cpu().numpy():
            gait_type = GaitType(gait_idx)
            names.append(GAIT_PARAMS[gait_type].name)
        return names
