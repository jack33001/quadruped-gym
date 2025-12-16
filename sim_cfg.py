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


##
# Get absolute path to robot USD
##
QUADRUPED_GYM_DIR = os.path.dirname(os.path.abspath(__file__))
QUADRUPED_USD_PATH = os.path.join(QUADRUPED_GYM_DIR, "Quadruped URDF", "QuadrupedUSD.usd")


##
# Default joint positions matching Genesis config
##
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
# Custom termination functions
##

def base_height_below_threshold(env, minimum_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate if base height above local terrain drops below minimum threshold.
    
    Uses the terrain's env_origins to get the local ground height for each environment.
    """
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    
    # Get terrain origin heights for each environment
    if hasattr(env.scene, 'terrain') and env.scene.terrain is not None:
        terrain = env.scene.terrain
        if hasattr(terrain, 'env_origins'):
            # env_origins contains (x, y, z) spawn position for each env
            # The z component is the terrain height at that location
            terrain_height = terrain.env_origins[:, 2]
            height_above_terrain = root_pos[:, 2] - terrain_height
            return height_above_terrain < minimum_height
    
    # Fallback to absolute height
    return root_pos[:, 2] < minimum_height


def left_terrain_cell(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate if robot leaves its assigned terrain cell.
    
    Each environment is assigned to a terrain cell. If the robot walks too far
    from the cell center, terminate the episode.
    """
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    
    # Get terrain cell size and origins
    if hasattr(env.scene, 'terrain') and env.scene.terrain is not None:
        terrain = env.scene.terrain
        if hasattr(terrain, 'env_origins') and hasattr(terrain, 'cfg'):
            env_origins = terrain.env_origins
            
            # Get terrain cell size from config
            terrain_cfg = terrain.cfg
            if hasattr(terrain_cfg, 'terrain_generator') and terrain_cfg.terrain_generator is not None:
                cell_size = terrain_cfg.terrain_generator.size
                half_size_x = cell_size[0] / 2.0 - 0.5  # Small margin
                half_size_y = cell_size[1] / 2.0 - 0.5
                
                # Compute distance from cell center in xy plane
                dx = torch.abs(root_pos[:, 0] - env_origins[:, 0])
                dy = torch.abs(root_pos[:, 1] - env_origins[:, 1])
                
                # Check if outside cell bounds
                outside_x = dx > half_size_x
                outside_y = dy > half_size_y
                
                return outside_x | outside_y
    
    # Fallback: never terminate for this reason
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


##
# Custom reward functions
##

def foot_slip_penalty(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float = 0.1) -> torch.Tensor:
    """Penalize feet sliding on ground while in contact.
    
    Computes the velocity of feet that are in contact with the ground.
    Returns the sum of squared xy velocities for feet in contact.
    """
    # Get contact sensor
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # Get asset for body velocities
    asset = env.scene[asset_cfg.name]
    
    # Get contact forces (num_envs, num_bodies, 3)
    contact_forces = contact_sensor.data.net_forces_w
    num_contact_bodies = contact_forces.shape[1]
    
    # Check which feet are in contact (force magnitude > threshold)
    in_contact = torch.norm(contact_forces, dim=-1) > threshold  # (num_envs, num_bodies)
    
    # Get linear velocities of all bodies in world frame
    body_vel = asset.data.body_lin_vel_w  # (num_envs, num_bodies, 3)
    
    # Use velocities for the contact sensor bodies (last N bodies typically are feet)
    # Take the last num_contact_bodies from body velocities
    if body_vel.shape[1] >= num_contact_bodies:
        foot_vel = body_vel[:, -num_contact_bodies:, :]  # (num_envs, num_feet, 3)
    else:
        foot_vel = body_vel
    
    # Ensure shapes match
    if foot_vel.shape[1] != num_contact_bodies:
        # Fallback: just use all available velocities up to contact bodies count
        foot_vel = body_vel[:, :num_contact_bodies, :]
    
    # Compute xy velocity magnitude squared for feet in contact
    foot_vel_xy_sq = foot_vel[:, :, 0]**2 + foot_vel[:, :, 1]**2  # (num_envs, num_feet)
    
    # Only penalize feet that are in contact
    slip_penalty = torch.sum(foot_vel_xy_sq * in_contact.float(), dim=-1)  # (num_envs,)
    
    return slip_penalty


def distance_traveled_reward(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for distance traveled from spawn point.
    
    Encourages robot to walk far, even if it eventually leaves the terrain cell.
    """
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    
    # Get spawn origins
    if hasattr(env.scene, 'terrain') and env.scene.terrain is not None:
        terrain = env.scene.terrain
        if hasattr(terrain, 'env_origins'):
            env_origins = terrain.env_origins
            
            # Compute xy distance from spawn
            dx = root_pos[:, 0] - env_origins[:, 0]
            dy = root_pos[:, 1] - env_origins[:, 1]
            distance = torch.sqrt(dx**2 + dy**2)
            
            return distance
    
    # Fallback: no reward
    return torch.zeros(env.num_envs, device=env.device)


def leg_contact_penalty(env, thigh_sensor_cfg: SceneEntityCfg, shin_sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize thigh and shin contacts with ground.
    
    Returns sum of contact force magnitudes above threshold for thighs and shins.
    """
    total_penalty = torch.zeros(env.num_envs, device=env.device)
    
    # Check thigh contacts
    thigh_sensor = env.scene.sensors[thigh_sensor_cfg.name]
    thigh_forces = thigh_sensor.data.net_forces_w
    thigh_force_mags = torch.norm(thigh_forces, dim=-1)  # (num_envs, num_bodies)
    thigh_penalty = torch.sum(torch.clamp(thigh_force_mags - threshold, min=0.0), dim=-1)
    total_penalty += thigh_penalty
    
    # Check shin contacts
    shin_sensor = env.scene.sensors[shin_sensor_cfg.name]
    shin_forces = shin_sensor.data.net_forces_w
    shin_force_mags = torch.norm(shin_forces, dim=-1)
    shin_penalty = torch.sum(torch.clamp(shin_force_mags - threshold, min=0.0), dim=-1)
    total_penalty += shin_penalty
    
    return total_penalty


def heading_tracking_reward(env, asset_cfg: SceneEntityCfg, std: float = 0.25) -> torch.Tensor:
    """Reward for maintaining the initial heading (yaw) orientation.
    
    Uses exponential kernel to reward small heading errors.
    The target heading is stored per environment when reset occurs.
    """
    asset = env.scene[asset_cfg.name]
    
    # Get current quaternion (w, x, y, z)
    quat = asset.data.root_quat_w
    
    # Extract yaw from quaternion
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    current_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    # Get target heading (stored during reset)
    if not hasattr(env, '_target_heading'):
        env._target_heading = current_yaw.clone()
    
    # Compute heading error (wrapped to [-pi, pi])
    heading_error = current_yaw - env._target_heading
    heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
    
    # Exponential reward kernel
    reward = torch.exp(-heading_error.pow(2) / (std * std))
    
    return reward


def foot_air_time_reward(env, sensor_cfg: SceneEntityCfg, threshold: float = 0.25) -> torch.Tensor:
    """Reward feet for being in the air, with diminishing returns after threshold.
    
    Uses the formula (threshold - t_air) which:
    - Gives positive reward when t_air < threshold (encourages longer steps)
    - Gives negative reward when t_air > threshold (discourages excessively long steps)
    - Peak reward at t_air = 0 transitioning to penalty after threshold
    
    This encourages moderate-duration steps rather than very short shuffling
    or excessively long hang times.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    # Get air time for each foot (time since last contact)
    # air_time is (num_envs, num_feet)
    air_time = contact_sensor.data.current_air_time
    
    # Compute reward: (threshold - air_time) for feet currently in air
    # Positive when air_time < threshold, negative when > threshold
    air_reward = threshold - air_time
    
    # Only count feet that are actually in the air (air_time > 0)
    in_air = air_time > 0.0
    
    # Sum rewards across all feet, only for feet in air
    total_reward = torch.sum(air_reward * in_air.float(), dim=-1)
    
    return total_reward


def foot_step_height_reward(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, target_height: float = 0.05) -> torch.Tensor:
    """Reward feet for lifting higher during swing phase.
    
    Tracks the position where each foot lifted off the ground and rewards
    height gained relative to that liftoff point. Uses exponential kernel
    centered at target_height.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    
    # Get contact forces to determine which feet are in air
    contact_forces = contact_sensor.data.net_forces_w
    num_feet = contact_forces.shape[1]
    in_contact = torch.norm(contact_forces, dim=-1) > 1.0  # (num_envs, num_feet)
    in_air = ~in_contact
    
    # Get foot positions in world frame
    body_pos = asset.data.body_pos_w  # (num_envs, num_bodies, 3)
    
    # Get foot z positions (assuming feet are last N bodies)
    if body_pos.shape[1] >= num_feet:
        foot_pos_z = body_pos[:, -num_feet:, 2]  # (num_envs, num_feet)
    else:
        foot_pos_z = body_pos[:, :num_feet, 2]
    
    # Initialize liftoff tracking buffers if they don't exist
    if not hasattr(env, '_foot_liftoff_z'):
        env._foot_liftoff_z = foot_pos_z.clone()
        env._foot_was_in_contact = in_contact.clone()
    
    # Detect feet that just lifted off (were in contact, now in air)
    just_lifted = env._foot_was_in_contact & in_air
    
    # Update liftoff positions for feet that just lifted off
    env._foot_liftoff_z = torch.where(just_lifted, foot_pos_z, env._foot_liftoff_z)
    
    # Compute height above liftoff point
    height_above_liftoff = foot_pos_z - env._foot_liftoff_z  # (num_envs, num_feet)
    
    # Reward for being close to target height above liftoff (only for feet in air)
    height_error = height_above_liftoff - target_height
    height_reward = torch.exp(-height_error.pow(2) / (0.02 * 0.02))  # std = 2cm
    
    # Only reward feet that are in the air
    total_reward = torch.sum(height_reward * in_air.float(), dim=-1)
    
    # Update contact state for next step
    env._foot_was_in_contact = in_contact.clone()
    
    return total_reward


def default_pose_reward(env, asset_cfg: SceneEntityCfg, std: float = 0.5) -> torch.Tensor:
    """Reward for joint positions being close to default pose.
    
    Uses exponential kernel on the L2 norm of joint position error
    relative to default positions.
    """
    asset = env.scene[asset_cfg.name]
    
    # Get current joint positions relative to default
    joint_pos_error = asset.data.joint_pos - asset.data.default_joint_pos
    
    # Compute L2 norm of error across all joints
    error_norm = torch.norm(joint_pos_error, dim=-1)
    
    # Exponential reward kernel
    reward = torch.exp(-error_norm.pow(2) / (std * std))
    
    return reward


def store_initial_heading(env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    """Event function to store initial heading after reset.
    
    Call this as a reset event to capture the randomized initial heading.
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    current_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    # Initialize target heading buffer if it doesn't exist
    if not hasattr(env, '_target_heading'):
        env._target_heading = torch.zeros(env.num_envs, device=env.device)
    
    # Only update heading for environments that were reset
    env._target_heading[env_ids] = current_yaw[env_ids].clone()


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
# Terrain Configuration
##

# Sub-terrains: each entry becomes a COLUMN in the terrain grid
# Difficulty varies across ROWS (controlled by difficulty_range)
# At difficulty 0 (row 0), height ranges start at 0 = completely flat
# At difficulty 1 (max row), height ranges reach their maximum
TERRAIN_SUB_TERRAINS = {
    # Column 0: Random grid boxes - height increases with difficulty
    "random_grid": MeshRandomGridTerrainCfg(
        proportion=1.0,
        grid_width=0.15,
        grid_height_range=(0.0, 0.12),  # Starts flat at difficulty 0, up to 12cm at difficulty 1
        platform_width=1.5,
    ),
    # Column 1: Pyramid stairs up - step height increases with difficulty
    "pyramid_stairs_up": MeshPyramidStairsTerrainCfg(
        proportion=1.0,
        step_height_range=(0.0, 0.12),  # Starts flat at difficulty 0, up to 12cm steps at difficulty 1
        step_width=0.3,
        platform_width=1.5,
        border_width=0.1,
        holes=False,
    ),
    # Column 2: Inverted pyramid stairs (going down into pit)
    "pyramid_stairs_down": MeshInvertedPyramidStairsTerrainCfg(
        proportion=1.0,
        step_height_range=(0.0, 0.12),  # Starts flat at difficulty 0, up to 12cm depth at difficulty 1
        step_width=0.3,
        platform_width=1.5,
        border_width=0.1,
        holes=False,
    ),
}


TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,  # Increased from 0.25 to make a more substantial border
    border_height=0.0,  # Negative = extends ABOVE ground (0.5m tall walls)
    num_rows=6,   # 6 difficulty levels (row 0 = flat, rows 1-5 = increasing difficulty)
    num_cols=3,   # 3 terrain types (grid, stairs up, stairs down)
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains=TERRAIN_SUB_TERRAINS,
    curriculum=True,  # Row 0 = easiest (diff=0 = flat), Row 5 = hardest (diff=1)
    difficulty_range=(0.0, 1.0),
)


##
# Scene Configuration
##

@configclass
class QuadrupedSceneCfg(InteractiveSceneCfg):
    """Configuration for quadruped scene."""

    # Terrain instead of ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="generator",
        terrain_generator=TERRAIN_CFG,
        max_init_terrain_level=0,  # All robots start on row 0 (easiest)
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = QUADRUPED_CFG

    # IMU sensor on torso
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Quadruped/Torso/Torso",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )

    # Contact sensor for detecting illegal body contacts (torso only)
    torso_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Quadruped/Torso/Torso",
        history_length=3,
        track_air_time=False,
        update_period=0.0,
    )
    
    # Contact sensor for thighs - illegal to touch ground
    thigh_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Quadruped/Torso/.*_Thigh",
        history_length=3,
        track_air_time=False,
        update_period=0.0,
    )

    # Contact sensor for shins - illegal to touch ground
    shin_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Quadruped/Torso/.*_Shin",
        history_length=3,
        track_air_time=False,
        update_period=0.0,
    )

    # Contact sensor for feet - used for foot slip detection (NOT illegal contact)
    foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Quadruped/Torso/.*_Foot",
        history_length=3,
        track_air_time=True,
        update_period=0.0,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
