"""
RL configuration for quadruped locomotion training.

Contains rewards, terminations, observations, actions, commands, events,
training hyperparameters, and curriculum settings.
"""
import math
from dataclasses import dataclass, field

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from sim_cfg import (
    FlatGroundSceneCfg,
    imu_angular_velocity,
    imu_projected_gravity,
    foot_slip_penalty,
    leg_contact_penalty,
    store_initial_heading,
    joint_torque_penalty,
    joint_jerk_penalty,
    ride_height_reward,
    base_orientation_reward,
    hip_position_reward,
)


##
# Observation Configuration
##

@configclass
class ObservationsCfg:
    """Observation configuration using IMU sensor data."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy - using IMU data for realistic sim-to-real."""

        # Velocity commands
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )

        # Gravity projection from IMU orientation
        projected_gravity = ObservationTermCfg(
            func=imu_projected_gravity,
            params={"sensor_cfg": SceneEntityCfg("imu")},
        )

        # Angular velocity from IMU
        base_ang_vel = ObservationTermCfg(
            func=imu_angular_velocity,
            params={"sensor_cfg": SceneEntityCfg("imu")},
            scale=0.25,
        )

        # Joint positions (relative to default)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, scale=1.0)

        # Joint velocities
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, scale=0.05)

        # Last actions
        last_action = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Action Configuration
##

@configclass
class ActionsCfg:
    """Action configuration."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


##
# Command Configuration
##

@configclass
class CommandsCfg:
    """Command configuration."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 2.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


##
# Event Configuration
##

@configclass
class EventCfg:
    """Configuration for events/randomization."""

    # Reset robot on terrain
    reset_robot = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0,0), "y": (0,0), "yaw": (0,0)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )

    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # Store initial heading after reset for heading tracking reward
    store_heading = EventTermCfg(
        func=store_initial_heading,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


##
# Reward Configuration
##

@configclass
class RewardsCfg:
    """Reward configuration with grouped rewards.
    
    Groups:
    - Efficiency: joint_jerk, joint_torque, action_smoothness
    - Velocity tracking: velocity_tracking
    - Gait scheduler tracking: foot_contact_tracking, step_height_tracking (computed in wrapper)
    - Stability: foot_slip, body_rates, base_orientation, ride_height, hip_position
    
    Weights are set via TrainCfg.reward_weights.
    """

    # Efficiency group
    joint_jerk = RewardTermCfg(
        func=joint_jerk_penalty,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_torque = RewardTermCfg(
        func=joint_torque_penalty,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    action_smoothness = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=1.0,
    )

    # Velocity tracking group
    velocity_tracking = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    # Stability group
    foot_slip = RewardTermCfg(
        func=foot_slip_penalty,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("foot_contact"),
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.1,
        },
    )

    body_rates = RewardTermCfg(
        func=mdp.ang_vel_xy_l2,
        weight=1.0,
    )

    base_orientation = RewardTermCfg(
        func=base_orientation_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.25,
        },
    )

    ride_height = RewardTermCfg(
        func=ride_height_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.22,
            "std": 0.05,
        },
    )

    hip_position = RewardTermCfg(
        func=hip_position_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.3,
        },
    )

    leg_collision = RewardTermCfg(
        func=leg_contact_penalty,
        weight=1.0,
        params={
            "thigh_sensor_cfg": SceneEntityCfg("thigh_contact"),
            "shin_sensor_cfg": SceneEntityCfg("shin_contact"),
            "threshold": 1.0,
        },
    )


##
# Termination Configuration
##

@configclass
class TerminationsCfg:
    """Termination configuration."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(45.0)},
    )

    # Terminate if torso contacts the ground
    torso_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("torso_contact"),
            "threshold": 10.0,
        },
    )


##
# Main Environment Configuration
##

@configclass
class QuadrupedEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for quadruped locomotion environment."""

    scene: FlatGroundSceneCfg = FlatGroundSceneCfg(num_envs=8192, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        self.sim.dt = 0.02
        self.sim.render_interval = 1
        self.decimation = 1
        self.episode_length_s = 10.0
        self.viewer.eye = (8.0, 8.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@dataclass
class RewardWeightsCfg:
    """Grouped reward weights with master and sub-weights."""
    
    # Master group weights
    efficiency_weight: float = .001
    velocity_weight: float = 1.0
    gait_tracking_weight: float = .005
    stability_weight: float = 1.0
    
    # Efficiency sub-weights
    joint_jerk: float = -0.0001
    joint_torque: float = -0.00001
    action_smoothness: float = -0.001
    
    # Velocity tracking sub-weights
    velocity_tracking: float = 2.0
    
    # Gait scheduler tracking sub-weights (applied in wrapper)
    foot_contact_tracking: float = 0.06
    step_height_tracking: float = 0.2
    
    # Stability sub-weights
    foot_slip: float = -0.05
    body_rates: float = -0.05
    base_orientation: float = 0.2
    ride_height: float = 0.3
    hip_position: float = 0.5
    leg_collision: float = -0.1


##
# Training Configuration
##

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
