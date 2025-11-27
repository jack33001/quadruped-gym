"""
Configuration for quadruped locomotion training with Isaac Lab.
"""
import math
import os
from dataclasses import dataclass, field

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp


##
# Get absolute path to robot USD
##
QUADRUPED_GYM_DIR = os.path.dirname(os.path.abspath(__file__))
QUADRUPED_USD_PATH = os.path.join(QUADRUPED_GYM_DIR, "Quadruped URDF", "QuadrupedUSD.usd")


##
# Default joint positions matching Genesis config
##
DEFAULT_JOINT_ANGLES = {
    "Front_Right_Hip": -0.7,
    "Front_Right_Knee": 0.0,
    "Front_Left_Hip": 0.7,
    "Front_Left_Knee": 0.0,
    "Rear_Right_Hip": 0.7,
    "Rear_Right_Knee": 0.0,
    "Rear_Left_Hip": -0.7,
    "Rear_Left_Knee": 0.0,
}


##
# Custom observation functions for IMU data
##

def imu_angular_velocity(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get angular velocity from IMU sensor (in body frame)."""
    imu_sensor = env.scene.sensors[sensor_cfg.name]
    return imu_sensor.data.ang_vel_b


def imu_projected_gravity(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Compute projected gravity from IMU orientation."""
    imu_sensor = env.scene.sensors[sensor_cfg.name]
    quat = imu_sensor.data.quat_w  # (num_envs, 4) in [w, x, y, z] format
    
    # Rotate gravity into body frame using quaternion
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Compute gravity in body frame (inverse rotation)
    gx = 2 * (x * z - w * y)
    gy = 2 * (y * z + w * x)
    gz = w * w - x * x - y * y + z * z
    
    return torch.stack([gx, gy, gz], dim=-1)


##
# Robot Configuration
##

QUADRUPED_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=QUADRUPED_USD_PATH,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=False,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.1,
            angular_damping=0.1,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=1.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.28),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=DEFAULT_JOINT_ANGLES,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=8.0,
            velocity_limit=10.0,
            stiffness=20.0,
            damping=0.5,
        ),
    },
)


##
# Scene Configuration
##

@configclass
class QuadrupedSceneCfg(InteractiveSceneCfg):
    """Configuration for quadruped scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    robot: ArticulationCfg = QUADRUPED_CFG

    # IMU sensor on torso - using ImuCfg (not ImuSensorCfg)
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Quadruped/Torso/Torso",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )

    # Contact sensor for detecting illegal body contacts
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Quadruped/Torso/.*",
        history_length=3,
        track_air_time=False,
        update_period=0.0,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
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

    reset_robot = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
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


##
# Reward Configuration
##

@configclass
class RewardsCfg:
    """Reward configuration matching Genesis scales."""

    # Reward for matching commanded forward/lateral velocity
    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=20.0,
        params={"command_name": "base_velocity", "std": 0.05},
    )

    # Reward for matching commanded turning rate
    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp,
        weight=0.25,
        params={"command_name": "base_velocity", "std": 0.05},
    )

    # Punish roll/pitch angular velocity (wobbling)
    ang_vel_xy_l2 = RewardTermCfg(
        func=mdp.ang_vel_xy_l2,
        weight=-.005,
    )

    # Punish vertical bouncing
    lin_vel_z_l2 = RewardTermCfg(
        func=mdp.lin_vel_z_l2,
        weight=-.25,
    )

    # Punish jerky/rapid action changes
    action_rate_l2 = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.0005,
    )

    # Punish tilting away from upright orientation
    flat_orientation_l2 = RewardTermCfg(
        func=mdp.flat_orientation_l2,
        weight=-0.25,
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

    # Terminate if body, thighs, or shins contact the ground
    # Threshold increased to prevent false positives during initial settling
    illegal_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
                #"Torso",
                #"Front_Right_Thigh",
                #"Front_Left_Thigh",
                #"Rear_Right_Thigh",
                #"Rear_Left_Thigh",
                #"Front_Right_Shin",
                #"Front_Left_Shin",
                #"Rear_Right_Shin",
                #"Rear_Left_Shin",
            ]),
            "threshold": 10.0,  # Increased from 1.0 - requires significant contact force
        },
    )


##
# Main Environment Configuration
##

@configclass
class QuadrupedEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for quadruped locomotion environment."""

    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(num_envs=16384, env_spacing=2.0)
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
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


##
# Training Configuration
##

@dataclass
class TrainCfg:
    """RSL-RL training configuration."""

    experiment_name: str = "quadruped_walking"
    run_name: str = ""
    max_iterations: int =150
    save_interval: int = 50
    log_interval: int = 1
    seed: int = 42
    num_steps_per_env: int = 24
    empirical_normalization: bool = False

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

    # Curriculum learning configuration
    # Iteration thresholds for each curriculum stage (stage activates at this iteration)
    curriculum_thresholds: list = field(default_factory=lambda: [0, 50, 100])

    # Reward weights for each curriculum stage
    # Keys must match reward term names in RewardsCfg
    curriculum_rewards: dict = field(default_factory=lambda: {
        "track_lin_vel_xy_exp": [20.0, 20.0, 20.0],
        "track_ang_vel_z_exp": [0.25, 0.5, 0.5],
        "ang_vel_xy_l2": [-0.005, -0.01, -0.025],
        "lin_vel_z_l2": [-0.25, -0.25, -0.25],
        "action_rate_l2": [-0.0005, -0.0005, -0.0005],
        "flat_orientation_l2": [0.0, -0.05, -0.25],
    })

    # Command ranges for each curriculum stage
    # Format: {key: [lin_vel_x_min, lin_vel_x_max, lin_vel_y_min, lin_vel_y_max, ang_vel_z_min, ang_vel_z_max]}
    curriculum_commands: dict = field(default_factory=lambda: {
        "lin_vel_x": [(0.5, 1.5), (0.5, 1.5), (0.5, 1.5)],
        "lin_vel_y": [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        "ang_vel_z": [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
    })