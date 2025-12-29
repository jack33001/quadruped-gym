"""
Training and evaluation configuration dataclasses.

These are pure Python dataclasses that can be imported before SimulationApp
is instantiated, unlike the Isaac Lab @configclass decorated classes.
"""
from dataclasses import dataclass, field


@dataclass
class SensorCfg:
    """Sensor thresholds and observation scaling factors."""
    
    foot_contact_threshold: float = 1.0
    torso_contact_threshold: float = 10.0
    leg_contact_threshold: float = 1.0
    slip_velocity_threshold: float = 0.1
    
    angular_velocity_scale: float = 0.25
    joint_velocity_scale: float = 0.05
    
    obs_clip_value: float = 100.0
    action_clip_value: float = 5.0
    
    velocity_deadzone: float = 0.3


@dataclass
class EvalCfg:
    """Configuration for policy evaluation."""
    
    headless: bool = False
    record_video: bool = True
    video_fps: int = 30
    evaluation_duration: float = 20.0
    
    camera_height: float = 3.0
    camera_distance: float = 5.0
    
    camera_pan_duration: float = 10.0
    camera_pan_height: float = 5.0
    camera_look_ahead: float = 15.0
    camera_target_height: float = 0.0
    
    spotlight_height: float = 15.0
    spotlight_intensity: float = 5000.0
    
    flat_ground_num_envs: int = 16


@dataclass
class RewardWeightsCfg:
    """Grouped reward weights with master and sub-weights."""
    
    efficiency_weight: float = .001
    velocity_weight: float = 1.0
    gait_tracking_weight: float = .005
    stability_weight: float = 1.0
    
    joint_jerk: float = -0.0001
    joint_torque: float = -0.00001
    action_smoothness: float = -0.001
    
    velocity_tracking: float = 2.0
    
    foot_contact_tracking: float = 0.1
    step_height_tracking: float = 0.2
    
    foot_slip: float = -0.1
    body_rates: float = -0.05
    base_orientation: float = 0.2
    ride_height: float = 0.3
    hip_position: float = 0.5
    leg_collision: float = -0.1


@dataclass
class TrainCfg:
    """RSL-RL training configuration."""

    experiment_name: str = "quadruped_gait"
    run_name: str = ""
    max_iterations: int = 250
    save_interval: int = 50
    log_interval: int = 1
    seed: int = 42
    num_steps_per_env: int = 24
    empirical_normalization: bool = False
    headless: bool = True

    obs_groups: dict = field(default_factory=lambda: {
        "policy": ["policy"],
        "critic": ["policy"],
    })

    algorithm: dict = field(default_factory=lambda: {
        "class_name": "PPO",
        "clip_param": 0.2,
        "desired_kl": 0.01,
        "entropy_coef": 0.01,
        "gamma": 0.99,
        "lam": 0.95,
        "learning_rate": 3e-4,
        "max_grad_norm": 1.0,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "schedule": "adaptive",
        "use_clipped_value_loss": True,
        "value_loss_coef": 1.0,
    })

    policy: dict = field(default_factory=lambda: {
        "class_name": "ActorCritic",
        "activation": "lrelu",
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "init_noise_std": 0.5,
    })

    reward_weights: RewardWeightsCfg = field(default_factory=RewardWeightsCfg)
    sensor: SensorCfg = field(default_factory=SensorCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
