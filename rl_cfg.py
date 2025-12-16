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
    QuadrupedSceneCfg,
    imu_angular_velocity,
    imu_projected_gravity,
    base_height_below_threshold,
    foot_slip_penalty,
    distance_traveled_reward,
    leg_contact_penalty,
    heading_tracking_reward,
    store_initial_heading,
    foot_air_time_reward,
    foot_step_height_reward,
    default_pose_reward,
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

        # Gravity projection from IMU orientation
        projected_gravity = ObservationTermCfg(
            func=imu_projected_gravity,
            params={"sensor_cfg": SceneEntityCfg("imu")},
        )

        # Velocity commands
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
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
            lin_vel_x=(0.5, 1.5),
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
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14159, 3.14159)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.0, 0.0),
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
    """Reward configuration.
    
    Note: Initial weights are placeholders. Actual weights are set by the
    curriculum system in train.py from TrainCfg.curriculum_rewards.
    """

    # Reward for matching commanded forward/lateral velocity
    cmd_lin_vel_tracking = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.0,  # Set by curriculum
        params={"command_name": "base_velocity", "std": 0.05},
    )

    # Reward for matching commanded yaw rate (turning)
    cmd_yaw_rate_tracking = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp,
        weight=0.0,  # Set by curriculum
        params={"command_name": "base_velocity", "std": 0.05},
    )

    # Reward for maintaining initial heading orientation
    heading_tracking = RewardTermCfg(
        func=heading_tracking_reward,
        weight=0.0,  # Set by curriculum
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.25,
        },
    )

    # Penalize roll and pitch angular velocity (wobbling)
    roll_pitch_rate_penalty = RewardTermCfg(
        func=mdp.ang_vel_xy_l2,
        weight=0.0,  # Set by curriculum
    )

    # Penalize vertical velocity (bouncing)
    vertical_vel_penalty = RewardTermCfg(
        func=mdp.lin_vel_z_l2,
        weight=0.0,  # Set by curriculum
    )

    # Penalize jerky/rapid action changes
    action_smoothness_penalty = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=0.0,  # Set by curriculum
    )

    # Penalize roll and pitch deviation from upright
    roll_pitch_orientation_penalty = RewardTermCfg(
        func=mdp.flat_orientation_l2,
        weight=0.0,  # Set by curriculum
    )

    # Penalize feet sliding on ground while in contact
    foot_slip_penalty = RewardTermCfg(
        func=foot_slip_penalty,
        weight=0.0,  # Set by curriculum
        params={
            "sensor_cfg": SceneEntityCfg("foot_contact"),
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.1,
        },
    )

    # Reward for distance traveled (incentivizes walking far)
    distance_traveled_reward = RewardTermCfg(
        func=distance_traveled_reward,
        weight=0.0,  # Set by curriculum
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Penalize thigh and shin contacts with ground
    leg_collision_penalty = RewardTermCfg(
        func=leg_contact_penalty,
        weight=0.0,  # Set by curriculum (should be negative)
        params={
            "thigh_sensor_cfg": SceneEntityCfg("thigh_contact"),
            "shin_sensor_cfg": SceneEntityCfg("shin_contact"),
            "threshold": 1.0,
        },
    )

    # Reward for foot air time (encourages longer steps, discourages shuffling)
    foot_air_time = RewardTermCfg(
        func=foot_air_time_reward,
        weight=0.0,  # Set by curriculum
        params={
            "sensor_cfg": SceneEntityCfg("foot_contact"),
            "threshold": 0.0,
        },
    )

    # Reward for lifting feet higher during swing phase
    foot_step_height = RewardTermCfg(
        func=foot_step_height_reward,
        weight=0.0,  # Set by curriculum
        params={
            "sensor_cfg": SceneEntityCfg("foot_contact"),
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.05,
        },
    )

    # Reward for staying close to default joint positions
    default_pose = RewardTermCfg(
        func=default_pose_reward,
        weight=0.0,  # Set by curriculum
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.5,
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

    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(num_envs=4096, env_spacing=2.5)
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
        
        # Increase PhysX GPU buffers for complex terrain collisions
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**24
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**22
        self.sim.physx.gpu_max_rigid_contact_count = 2**24
        self.sim.physx.gpu_max_rigid_patch_count = 2**22
        self.sim.physx.gpu_heap_capacity = 2**27
        self.sim.physx.gpu_temp_buffer_capacity = 2**25


##
# Training Configuration
##

@dataclass
class TrainCfg:
    """RSL-RL training configuration."""

    experiment_name: str = "quadruped_walking"
    run_name: str = ""
    max_iterations: int = 400
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
        "activation": "elu",
        "actor_hidden_dims": [256, 128, 64],
        "critic_hidden_dims": [256, 128, 64],
        "init_noise_std": 0.5,
    })

    # Unified curriculum configuration (terrain + rewards coupled)
    # 6 stages to match 6 terrain difficulty rows
    curriculum_num_stages: int = 6  # Number of curriculum stages
    curriculum_promote_threshold: float = 30.0  # Mean episode reward to advance
    curriculum_demote_threshold: float = 10.0   # Mean episode reward to retreat
    curriculum_update_freq: int = 10  # Iterations between curriculum updates
    curriculum_min_episodes: int = 50  # Min episodes at a level before promote/demote
    curriculum_cooldown_promote: int = 20  # Iterations to wait after promotion before next decision
    curriculum_cooldown_demote: int = 10   # Iterations to wait after demotion before next decision

    # Reward weights per curriculum stage (index = stage, 6 stages total)
    # Gradually increase tracking rewards and penalties across stages
    curriculum_rewards: dict = field(default_factory=lambda: {
        # Stage 0-5: Progressively stricter rewards
        "cmd_lin_vel_tracking": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        "cmd_yaw_rate_tracking": [0.025, 0.025, 0.05, 0.05, 0.05, 0.05],
        "heading_tracking": [-0.1, -.1, -0.1, -0.1, -0.1, -0.1],
        "roll_pitch_rate_penalty": [-0.005, -0.005, -0.01, -0.01, -0.025, -0.025],
        "roll_pitch_orientation_penalty": [-0.0, -0.0, -0.05, -0.05, -0.25, -0.25],
        "vertical_vel_penalty": [-0.25, -0.25, -0.25, -0.25, -0.25, -0.25],
        "action_smoothness_penalty": [-0.0005, -0.0005, -0.0005, -0.0005, -0.0005, -0.0005],
        "foot_slip_penalty": [0.0, 0.0, -0.005, -0.005, -0.02, -0.02],
        "distance_traveled_reward": [.5, .5, .5, .5, .5, .5],
        "leg_collision_penalty": [-.01, -.01, -.01, -.01, -.01, -.01],
        "foot_air_time": [0.0, 0.0, 0.05, 0.05, 0.1, 0.1],
        "foot_step_height": [0.0, 0.0, 0.05, 0.05, 0.1, 0.1],
        "default_pose": [0.01, 0.01, 0.05, 0.05, 0.1, 0.1],
    })


    # Command ranges per curriculum stage (6 stages)
    # Gradually increase velocity and turning demands
    curriculum_commands: dict = field(default_factory=lambda: {
        "lin_vel_x": [(0.3, 0.6), (0.3, 0.6), (0.3, 0.6), (0.3, 0.6), (0.3, 0.6), (0.3, 0.6)],
        "lin_vel_y": [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        "ang_vel_z": [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
    })
