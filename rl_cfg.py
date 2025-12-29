"""
RL configuration for quadruped locomotion training.

Contains rewards, terminations, observations, actions, commands, events,
and environment configuration using Isaac Lab @configclass decorators.
"""
import math
from dataclasses import field

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

from train_cfg import SensorCfg, EvalCfg, RewardWeightsCfg, TrainCfg


##
# Sensor and Scaling Configuration
##

SENSOR_CFG = SensorCfg()


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
            scale=SENSOR_CFG.angular_velocity_scale,
        )

        # Joint positions (relative to default)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, scale=1.0)

        # Joint velocities
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel, 
            scale=SENSOR_CFG.joint_velocity_scale
        )

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
        scale=0.75,
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
            "threshold": SENSOR_CFG.slip_velocity_threshold,
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
            "threshold": SENSOR_CFG.leg_contact_threshold,
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
            "threshold": SENSOR_CFG.torso_contact_threshold,
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
    
    sensor: SensorCfg = field(default_factory=SensorCfg)

    def __post_init__(self):
        """Post initialization."""
        self.sim.dt = 0.02
        self.sim.render_interval = 1
        self.decimation = 1
        self.episode_length_s = 10.0
        self.viewer.eye = (8.0, 8.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
