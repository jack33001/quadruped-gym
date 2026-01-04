"""
Simulation configuration for quadruped locomotion with Isaac Lab.

Contains robot definition, terrain configuration, scene setup, and sensors.
"""
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import (
    MeshPyramidStairsTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshRandomGridTerrainCfg,
)
from isaaclab.utils import configclass


QUADRUPED_GYM_DIR = os.path.dirname(os.path.abspath(__file__))
QUADRUPED_USD_PATH = os.path.join(QUADRUPED_GYM_DIR, "Quadruped URDF", "QuadrupedUSD.usd")


DEFAULT_JOINT_ANGLES = {
    "Front_Right_Hip": -0.6,
    "Front_Right_Knee": -0.2,
    "Front_Left_Hip": 0.6,
    "Front_Left_Knee": 0.2,
    "Rear_Right_Hip": 0.6,
    "Rear_Right_Knee": 0.2,
    "Rear_Left_Hip": -0.6,
    "Rear_Left_Knee": -0.2,
}

TARGET_BASE_HEIGHT = 0.22


##
# Shared sensor configurations
##

IMU_CFG = ImuCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Quadruped/Torso/Torso",
    update_period=0.0,
    history_length=1,
    debug_vis=False,
)

TORSO_CONTACT_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Quadruped/Torso/Torso",
    history_length=3,
    track_air_time=False,
    update_period=0.0,
)

THIGH_CONTACT_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Quadruped/Torso/.*_Thigh",
    history_length=3,
    track_air_time=False,
    update_period=0.0,
)

SHIN_CONTACT_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Quadruped/Torso/.*_Shin",
    history_length=3,
    track_air_time=False,
    update_period=0.0,
)

FOOT_CONTACT_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Quadruped/Torso/.*_Foot",
    history_length=3,
    track_air_time=True,
    update_period=0.0,
)

LIGHT_CFG = AssetBaseCfg(
    prim_path="/World/light",
    spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
)


##
# Utility functions
##

def compute_projected_gravity(quat: torch.Tensor) -> torch.Tensor:
    """Compute gravity vector projected into body frame from quaternion.
    
    Args:
        quat: Quaternion tensor of shape (num_envs, 4) in (w, x, y, z) order.
        
    Returns:
        Projected gravity tensor of shape (num_envs, 3).
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    gx = 2 * (x * z - w * y)
    gy = 2 * (y * z + w * x)
    gz = w * w - x * x - y * y + z * z
    
    return torch.stack([gx, gy, gz], dim=-1)


##
# Custom observation functions
##

def imu_angular_velocity(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get angular velocity from IMU sensor in body frame."""
    imu_sensor = env.scene.sensors[sensor_cfg.name]
    return imu_sensor.data.ang_vel_b


def imu_projected_gravity(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Compute gravity vector projected into body frame from IMU orientation."""
    imu_sensor = env.scene.sensors[sensor_cfg.name]
    return compute_projected_gravity(imu_sensor.data.quat_w)


def base_pos_z(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get base height (z position)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3]


def foot_contact(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Get binary foot contact state based on contact force threshold."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w
    contact_magnitude = torch.norm(contact_forces, dim=-1)
    return (contact_magnitude > threshold).float()


def applied_torque(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get applied joint torques."""
    asset = env.scene[asset_cfg.name]
    return asset.data.applied_torque


##
# Custom termination functions
##

def base_height_below_threshold(env, minimum_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate if base height above local terrain drops below minimum threshold."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    
    if hasattr(env.scene, 'terrain') and env.scene.terrain is not None:
        terrain = env.scene.terrain
        if hasattr(terrain, 'env_origins'):
            terrain_height = terrain.env_origins[:, 2]
            height_above_terrain = root_pos[:, 2] - terrain_height
            return height_above_terrain < minimum_height
    
    return root_pos[:, 2] < minimum_height


def left_terrain_cell(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate if robot leaves its assigned terrain cell."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    
    if hasattr(env.scene, 'terrain') and env.scene.terrain is not None:
        terrain = env.scene.terrain
        if hasattr(terrain, 'env_origins') and hasattr(terrain, 'cfg'):
            env_origins = terrain.env_origins
            
            terrain_cfg = terrain.cfg
            if hasattr(terrain_cfg, 'terrain_generator') and terrain_cfg.terrain_generator is not None:
                cell_size = terrain_cfg.terrain_generator.size
                half_size_x = cell_size[0] / 2.0 - 0.5
                half_size_y = cell_size[1] / 2.0 - 0.5
                
                dx = torch.abs(root_pos[:, 0] - env_origins[:, 0])
                dy = torch.abs(root_pos[:, 1] - env_origins[:, 1])
                
                outside_x = dx > half_size_x
                outside_y = dy > half_size_y
                
                return outside_x | outside_y
    
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


##
# Custom reward functions
##

def foot_slip_penalty(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float = 0.1) -> torch.Tensor:
    """Penalize feet sliding on ground while in contact.
    
    Args:
        env: The environment instance.
        sensor_cfg: Configuration for foot contact sensor.
        asset_cfg: Configuration for robot asset.
        threshold: Minimum contact force to consider foot in contact.
        
    Returns:
        Slip penalty tensor of shape (num_envs,).
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    
    contact_forces = contact_sensor.data.net_forces_w
    num_contact_bodies = contact_forces.shape[1]
    
    in_contact = torch.norm(contact_forces, dim=-1) > threshold
    
    body_vel = asset.data.body_lin_vel_w
    
    if body_vel.shape[1] >= num_contact_bodies:
        foot_vel = body_vel[:, -num_contact_bodies:, :]
    else:
        foot_vel = body_vel
    
    if foot_vel.shape[1] != num_contact_bodies:
        foot_vel = body_vel[:, :num_contact_bodies, :]
    
    foot_vel_xy_sq = foot_vel[:, :, 0]**2 + foot_vel[:, :, 1]**2
    
    slip_penalty = torch.sum(foot_vel_xy_sq * in_contact.float(), dim=-1)
    
    return slip_penalty


def leg_contact_penalty(env, thigh_sensor_cfg: SceneEntityCfg, shin_sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize thigh and shin contacts with ground.
    
    Args:
        env: The environment instance.
        thigh_sensor_cfg: Configuration for thigh contact sensor.
        shin_sensor_cfg: Configuration for shin contact sensor.
        threshold: Force threshold above which to penalize.
        
    Returns:
        Contact penalty tensor of shape (num_envs,).
    """
    total_penalty = torch.zeros(env.num_envs, device=env.device)
    
    thigh_sensor = env.scene.sensors[thigh_sensor_cfg.name]
    thigh_forces = thigh_sensor.data.net_forces_w
    thigh_force_mags = torch.norm(thigh_forces, dim=-1)
    thigh_penalty = torch.sum(torch.clamp(thigh_force_mags - threshold, min=0.0), dim=-1)
    total_penalty += thigh_penalty
    
    shin_sensor = env.scene.sensors[shin_sensor_cfg.name]
    shin_forces = shin_sensor.data.net_forces_w
    shin_force_mags = torch.norm(shin_forces, dim=-1)
    shin_penalty = torch.sum(torch.clamp(shin_force_mags - threshold, min=0.0), dim=-1)
    total_penalty += shin_penalty
    
    return total_penalty


def store_initial_heading(env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    """Event function to store initial heading after reset.
    
    Args:
        env: The environment instance.
        env_ids: Indices of environments being reset.
        asset_cfg: Configuration for robot asset.
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    current_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    if not hasattr(env, '_target_heading'):
        env._target_heading = torch.zeros(env.num_envs, device=env.device)
    
    env._target_heading[env_ids] = current_yaw[env_ids].clone()


def joint_torque_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize high joint torques (L2 norm).
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for robot asset.
        
    Returns:
        Torque penalty tensor of shape (num_envs,).
    """
    asset = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque
    return torch.sum(torques.pow(2), dim=-1)


def joint_jerk_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint acceleration changes (jerk).
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for robot asset.
        
    Returns:
        Jerk penalty tensor of shape (num_envs,).
    """
    asset = env.scene[asset_cfg.name]
    joint_acc = asset.data.joint_acc
    
    if not hasattr(env, '_prev_joint_acc'):
        env._prev_joint_acc = joint_acc.clone()
        return torch.zeros(env.num_envs, device=env.device)
    
    jerk = joint_acc - env._prev_joint_acc
    env._prev_joint_acc = joint_acc.clone()
    
    return torch.sum(jerk.pow(2), dim=-1)


def ride_height_reward(env, asset_cfg: SceneEntityCfg, target_height: float = 0.22, std: float = 0.05) -> torch.Tensor:
    """Reward for maintaining target base height above terrain.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for robot asset.
        target_height: Desired height above terrain.
        std: Standard deviation for exponential reward shaping.
        
    Returns:
        Height reward tensor of shape (num_envs,).
    """
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    
    terrain_height = torch.zeros(env.num_envs, device=env.device)
    if hasattr(env.scene, 'terrain') and env.scene.terrain is not None:
        terrain = env.scene.terrain
        if hasattr(terrain, 'env_origins'):
            terrain_height = terrain.env_origins[:, 2]
    
    height_above_terrain = root_pos[:, 2] - terrain_height
    height_error = height_above_terrain - target_height
    
    reward = torch.exp(-height_error.pow(2) / (std * std))
    
    return reward


def base_orientation_reward(env, asset_cfg: SceneEntityCfg, std: float = 0.25) -> torch.Tensor:
    """Reward for keeping base level (roll and pitch near zero).
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for robot asset.
        std: Standard deviation for exponential reward shaping.
        
    Returns:
        Orientation reward tensor of shape (num_envs,).
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    
    orientation_error = roll.pow(2) + pitch.pow(2)
    reward = torch.exp(-orientation_error / (std * std))
    
    return reward


def hip_position_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize hip joint angles deviating from default position.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for robot asset.
        
    Returns:
        Hip position penalty tensor of shape (num_envs,).
    """
    asset = env.scene[asset_cfg.name]
    
    joint_pos = asset.data.joint_pos
    default_pos = asset.data.default_joint_pos
    
    hip_indices = [0, 2, 4, 6]
    
    hip_error = torch.zeros(env.num_envs, device=env.device)
    for idx in hip_indices:
        if idx < joint_pos.shape[1]:
            hip_error += (joint_pos[:, idx] - default_pos[:, idx]).pow(2)
    
    return hip_error


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
# Scene Configurations
##

@configclass
class FlatGroundSceneCfg(InteractiveSceneCfg):
    """Configuration for flat ground training scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot: ArticulationCfg = QUADRUPED_CFG
    imu: ImuCfg = IMU_CFG
    torso_contact: ContactSensorCfg = TORSO_CONTACT_CFG
    thigh_contact: ContactSensorCfg = THIGH_CONTACT_CFG
    shin_contact: ContactSensorCfg = SHIN_CONTACT_CFG
    foot_contact: ContactSensorCfg = FOOT_CONTACT_CFG
    light: AssetBaseCfg = LIGHT_CFG


##
# Terrain Configuration
##

TERRAIN_SUB_TERRAINS = {
    "random_grid": MeshRandomGridTerrainCfg(
        proportion=1.0,
        grid_width=0.15,
        grid_height_range=(0.0, 0.12),
        platform_width=1.5,
    ),
    "pyramid_stairs_up": MeshPyramidStairsTerrainCfg(
        proportion=1.0,
        step_height_range=(0.0, 0.12),
        step_width=0.3,
        platform_width=1.5,
        border_width=0.1,
        holes=False,
    ),
    "pyramid_stairs_down": MeshInvertedPyramidStairsTerrainCfg(
        proportion=1.0,
        step_height_range=(0.0, 0.12),
        step_width=0.3,
        platform_width=1.5,
        border_width=0.1,
        holes=False,
    ),
}


TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    border_height=0.0,
    num_rows=6,
    num_cols=3,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains=TERRAIN_SUB_TERRAINS,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
)


@configclass
class QuadrupedSceneCfg(InteractiveSceneCfg):
    """Configuration for quadruped scene with procedural terrain."""

    terrain = TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="generator",
        terrain_generator=TERRAIN_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = QUADRUPED_CFG
    imu: ImuCfg = IMU_CFG
    torso_contact: ContactSensorCfg = TORSO_CONTACT_CFG
    thigh_contact: ContactSensorCfg = THIGH_CONTACT_CFG
    shin_contact: ContactSensorCfg = SHIN_CONTACT_CFG
    foot_contact: ContactSensorCfg = FOOT_CONTACT_CFG
    light: AssetBaseCfg = LIGHT_CFG
