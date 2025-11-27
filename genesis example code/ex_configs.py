"""
Configuration for quadruped locomotion training.
Centralized config following RSL-RL conventions.
"""
import numpy as np


class EnvConfig:
    """Environment configuration."""
    
    # Simulation
    dt = 0.02
    substeps = 6  # Increased for walking dynamics (was 4 for standing)
    
    # Environment
    num_envs = 24576  # Increased from 16384 - you have ~3.8GB headroom
    episode_length_s = 10.0  # Reduced from 25.0 - faster learning, more diverse resets
    num_actions = 8
    env_spacing = 1.0
    
    # Robot
    robot_file = 'Quadruped XML.xml'
    
    # Joint names (in control order)
    joint_names = [
        'Front Right Hip', 'Front Right Knee',
        'Front Left Hip', 'Front Left Knee',
        'Rear Right Hip', 'Rear Right Knee',
        'Rear Left Hip', 'Rear Left Knee',
    ]
    
    # Foot link names
    foot_link_names = [
        'Front Right Shin', 'Front Left Shin',
        'Rear Right Shin', 'Rear Left Shin',
    ]
    
    # Control
    action_scale = .5
    decimation = 1  # How many sim steps per action
    kp = 20.0  # Position gain
    kd = 0.5   # Velocity gain (called kv in your code)
    
    # Default joint positions (zero pose)
    default_joint_angles = np.array([-.7,0,.7,0,.7,0,-.7,0])  # Slightly crouched for walking
    
    # Physical properties
    base_init_pos = [0.0, 0.0, 0.22]
    base_init_quat = [1.0, 0.0, 0.0, 0.0]  # No rotation needed - STL is correctly oriented
    base_height_target = 0.218
    
    # Randomization
    base_height_range = [0.18, 0.26]  # Randomize ±4cm from nominal 0.22m
    
    # Termination
    termination_if_roll_greater_than = 45.0  # RELAXED - robot clearly can handle 45+ deg
    termination_if_pitch_greater_than = 45.0  # RELAXED - allow more dynamic motion
    
    # Viewer
    show_viewer = False
    max_fps = 30


class ObsConfig:
    """Observation configuration."""
    
    num_obs = 33  # Dimension of policy input
    
    # Observation components:
    # - Gravity (3)
    # - Commands (3) 
    # - Angular velocity (3)
    # - Joint positions (8)
    # - Joint velocities (8)
    # - Last actions (8)
    
    obs_scales = {
        'lin_vel': 2.0,
        'ang_vel': 0.25,
        'dof_pos': 1.0,
        'dof_vel': 0.05,
    }


class RewardConfig:
    """Reward function configuration."""
    
    # Gaussian tracking reward - MUCH STRICTER
    tracking_sigma = 0.05  # REDUCED from 0.1 - now 0.5 m/s error gives only ~0.0001 reward!
    
    # Reward scales
    scales = {
        'tracking_lin_vel_xy': 20.0,   # INCREASED - velocity tracking is PRIMARY objective
        'tracking_ang_vel_yaw': 0.25, # Track commanded yaw rate
        'ang_vel_pitch_roll': -5.0,   # INCREASED penalty - robot is too unstable in roll
        'lin_vel_z': -2.0,            # Penalize vertical velocity
        'base_height': -1.0,          # REDUCED - allow more height variation during gaits
        'action_rate': -0.1,          # REDUCED - allow more aggressive stepping
        'similar_to_default': -0.005, # REDUCED - walking requires deviation from default
    }
    
    # Targets
    base_height_target = 0.218


class CommandConfig:
    """Command sampling configuration."""
    
    num_commands = 3  # [vx, vy, yaw_rate]
    
    # Command ranges for WALKING task - FORCE forward motion
    lin_vel_x_range = [0.5, 1.5]    # Forward walking
    lin_vel_y_range = [0, 0]        # No lateral movement for now
    ang_vel_range = [0, 0]          # No turning for now
    

class TrainConfig:
    """RSL-RL training configuration."""
    
    # Runner
    experiment_name = "quadruped_walking"  # Changed from "quadruped_locomotion"
    run_name = ""
    max_iterations = 167  # Reduced from 250 (24576/16384 * 250 = ~375, but conservative)
    save_interval = 33  # Save every ~20% of training
    log_interval = 1
    
    # Algorithm (PPO)
    algorithm = {
        "class_name": "PPO",
        "clip_param": 0.2,
        "desired_kl": 0.01,
        "entropy_coef": 0.01,
        "gamma": 0.99,
        "lam": 0.95,
        "learning_rate": 1e-3,
        "max_grad_norm": 1.0,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,  # Keep at 4 - each minibatch ~147k samples (24576*24/4)
        "schedule": "adaptive",
        "use_clipped_value_loss": True,
        "value_loss_coef": 1.0,
    }
    
    # Policy (Actor-Critic)
    policy = {
        "class_name": "ActorCritic",
        "activation": "elu",
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "init_noise_std": 1.0,
    }
    
    # Observation groups
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }
    
    # Rollout
    num_steps_per_env = 24  # Transitions per env per iteration
    
    # Misc
    seed = 42
    empirical_normalization = None


class EvalConfig:
    """Evaluation configuration."""
    
    # Experiment directory to load from
    experiment_name = "quadruped_walking"  # Changed from hardcoded path
    
    # Checkpoint to load
    checkpoint = "model_166.pt"
    
    # Number of environments to visualize
    num_envs = 1
    
    # Number of episodes to evaluate
    num_episodes = 10


def get_cfg_dict(train_cfg: TrainConfig, env_cfg: EnvConfig):
    """Convert TrainConfig to RSL-RL config dictionary."""
    return {
        "algorithm": train_cfg.algorithm,
        "init_member_classes": {},
        "policy": train_cfg.policy,
        "runner": {
            "checkpoint": -1,
            "experiment_name": train_cfg.experiment_name,
            "load_run": -1,
            "log_interval": train_cfg.log_interval,
            "max_iterations": train_cfg.max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": train_cfg.run_name,
            "save_interval": train_cfg.save_interval,  # Make sure this is here
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": train_cfg.num_steps_per_env,
        "save_interval": train_cfg.save_interval,  # This is the key one for OnPolicyRunner
        "empirical_normalization": train_cfg.empirical_normalization,
        "obs_groups": train_cfg.obs_groups,
        "seed": train_cfg.seed,
    }
