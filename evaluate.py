"""
Evaluation script for trained quadruped policies.
Loads checkpoint and runs evaluation with optional video recording.
"""
import os
import sys

os.environ["PYTORCH_NVFUSER_DISABLE_FALLBACK"] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

# Configuration - edit these values as needed
HEADLESS = False  # Set to False to see visualization and capture frames
RECORD_VIDEO = True  # Enable video recording via frame capture
VIDEO_DIR = "logs/quadruped_walking/videos"
FRAMES_DIR = "logs/quadruped_walking/frames"
VIDEO_FPS = 30
NUM_ENVS = 36  # 6 rows x 3 cols = 18 subterrains, one robot per cell
EXPERIMENT_NAME = "quadruped_walking"
CHECKPOINT = "model_499.pt"

# Camera pan configuration
CAMERA_PAN_DURATION = 10.0  # Duration of camera pan in seconds
CAMERA_HEIGHT = 5.0  # Height above terrain
CAMERA_LOOK_AHEAD = 15.0  # Distance ahead to look (larger = shallower angle)
CAMERA_TARGET_HEIGHT = 0.0  # Height of look-at target above ground
SPOTLIGHT_HEIGHT = 15.0  # Height of overhead spotlight
SPOTLIGHT_INTENSITY = 5000.0  # Spotlight intensity

from isaaclab.app import AppLauncher

# Create app launcher
app_launcher = AppLauncher(
    headless=HEADLESS,
)
simulation_app = app_launcher.app

# Now we can import Isaac Lab modules
import pickle
import torch
import numpy as np
import shutil
import math

from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.modules import ActorCritic

from rl_cfg import QuadrupedEnvCfg
from quadruped_env import IsaacLabVecEnvWrapper


def setup_frame_capture():
    """Setup frame capture using viewport."""
    try:
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport is None:
            print("Warning: No active viewport found for frame capture")
            return None
        return viewport
    except Exception as e:
        print(f"Warning: Could not setup frame capture: {e}")
        return None


def capture_frame(viewport, frame_path: str):
    """Capture a single frame from the viewport and save to disk."""
    try:
        from omni.kit.viewport.utility import capture_viewport_to_file
        capture_viewport_to_file(viewport, frame_path)
        return True
    except Exception as e:
        return False


def frames_to_video(frames_dir: str, output_path: str, fps: int = 30):
    """Combine saved frames into a video using imageio."""
    try:
        import imageio
        
        # Get sorted list of frame files
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        if len(frame_files) == 0:
            print(f"No frames found in {frames_dir}")
            return False
        
        print(f"Combining {len(frame_files)} frames into video...")
        
        # Read frames and write video
        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            frame = imageio.imread(frame_path)
            frames.append(frame)
        
        imageio.mimwrite(output_path, frames, fps=fps)
        print(f"Video saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error creating video: {e}")
        return False


def get_terrain_bounds(env):
    """Get the bounds and cell info of the subterrain grid (excluding border)."""
    isaac_env = env.unwrapped
    
    # Default values
    cell_size = (8.0, 8.0)
    num_rows = 6
    num_cols = 3
    
    if hasattr(isaac_env.scene, 'terrain') and isaac_env.scene.terrain is not None:
        terrain = isaac_env.scene.terrain
        if hasattr(terrain, 'cfg') and hasattr(terrain.cfg, 'terrain_generator'):
            gen_cfg = terrain.cfg.terrain_generator
            if hasattr(gen_cfg, 'size'):
                cell_size = gen_cfg.size
            if hasattr(gen_cfg, 'num_rows'):
                num_rows = gen_cfg.num_rows
            if hasattr(gen_cfg, 'num_cols'):
                num_cols = gen_cfg.num_cols
    
    # Calculate the actual subterrain grid size (without border)
    grid_width = cell_size[0] * num_cols
    grid_height = cell_size[1] * num_rows
    
    # The subterrain grid is centered at origin
    return {
        "min_x": -grid_width / 2,
        "max_x": grid_width / 2,
        "min_y": -grid_height / 2,
        "max_y": grid_height / 2,
        "cell_size": cell_size,
        "num_rows": num_rows,
        "num_cols": num_cols,
    }


def setup_camera_pan(env, duration: float, height: float, look_ahead: float, target_height: float):
    """Setup camera pan parameters.
    
    Camera pans diagonally across the terrain grid, from the first subterrain
    cell to the last, keeping a fixed look direction to avoid rotation.
    """
    isaac_env = env.unwrapped
    terrain = isaac_env.scene.terrain
    
    # Get actual terrain cell origins from the terrain object
    if hasattr(terrain, 'terrain_origins'):
        origins = terrain.terrain_origins  # Shape: (num_rows, num_cols, 3)
        
        # First cell is at [0, 0], last cell is at [num_rows-1, num_cols-1]
        first_cell = origins[0, 0].cpu().numpy()
        last_cell = origins[-1, -1].cpu().numpy()
        
        first_cell_x = float(first_cell[0])
        first_cell_y = float(first_cell[1])
        last_cell_x = float(last_cell[0])
        last_cell_y = float(last_cell[1])
        
        print(f"First cell origin: ({first_cell_x:.1f}, {first_cell_y:.1f})")
        print(f"Last cell origin: ({last_cell_x:.1f}, {last_cell_y:.1f})")
    else:
        # Fallback to calculated positions
        bounds = get_terrain_bounds(env)
        cell_w = bounds["cell_size"][0]
        cell_h = bounds["cell_size"][1]
        first_cell_x = bounds["min_x"] + cell_w / 2
        first_cell_y = bounds["min_y"] + cell_h / 2
        last_cell_x = bounds["max_x"] - cell_w / 2
        last_cell_y = bounds["max_y"] - cell_h / 2
    
    # Calculate diagonal direction from first to last cell (normalized)
    dx = last_cell_x - first_cell_x
    dy = last_cell_y - first_cell_y
    dist = math.sqrt(dx*dx + dy*dy)
    if dist > 0.001:
        dir_x = dx / dist
        dir_y = dy / dist
    else:
        dir_x = 1.0
        dir_y = 0.0
    
    # Start directly above first cell
    start_pos = (first_cell_x, first_cell_y, height)
    
    # End position: stop before the last cell so it stays in view
    # Pull back by the look_ahead distance so the last cell is centered in view at the end
    end_pos = (
        last_cell_x - dir_x * look_ahead,
        last_cell_y - dir_y * look_ahead,
        height
    )
    
    return {
        "start_pos": start_pos,
        "end_pos": end_pos,
        "look_dir": (dir_x, dir_y),
        "look_ahead": look_ahead,
        "target_height": target_height,
        "duration": duration,
        "height": height,
    }


def set_camera_view(isaac_env, eye: tuple, target: tuple):
    """Set camera position and look-at target using Isaac Lab's sim API."""
    try:
        import numpy as np
        cam_eye = np.array(eye, dtype=float)
        cam_target = np.array(target, dtype=float)
        isaac_env.sim.set_camera_view(eye=cam_eye, target=cam_target)
        return True
    except Exception as e:
        if not hasattr(set_camera_view, '_error_count'):
            set_camera_view._error_count = 0
        set_camera_view._error_count += 1
        if set_camera_view._error_count <= 3:
            print(f"Warning: Could not set camera view: {e}")
        return False


def update_camera(isaac_env, pan_config: dict, progress: float):
    """Update camera position based on pan progress (0.0 to 1.0)."""
    start = pan_config["start_pos"]
    end = pan_config["end_pos"]
    look_dir = pan_config["look_dir"]
    look_ahead = pan_config["look_ahead"]
    target_height = pan_config.get("target_height", 0.0)
    
    # Interpolate eye position along the diagonal
    eye_x = start[0] + (end[0] - start[0]) * progress
    eye_y = start[1] + (end[1] - start[1]) * progress
    eye_z = pan_config["height"]
    
    # Target is a fixed distance ahead in the direction of travel
    target_x = eye_x + look_dir[0] * look_ahead
    target_y = eye_y + look_dir[1] * look_ahead
    target_z = target_height
    
    set_camera_view(
        isaac_env,
        eye=(eye_x, eye_y, eye_z),
        target=(target_x, target_y, target_z)
    )


def create_spotlight(isaac_env):
    """Create an overhead spotlight that can be moved during evaluation."""
    try:
        import omni.usd
        from pxr import UsdLux, Gf, Sdf
        
        stage = omni.usd.get_context().get_stage()
        
        # Create a sphere light for soft overhead illumination
        light_path = "/World/eval_spotlight"
        light_prim = stage.DefinePrim(light_path, "SphereLight")
        
        sphere_light = UsdLux.SphereLight(light_prim)
        sphere_light.CreateIntensityAttr(SPOTLIGHT_INTENSITY)
        sphere_light.CreateRadiusAttr(2.0)
        sphere_light.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.9))  # Slightly warm white
        
        # Enable shadow casting
        sphere_light.CreateEnableColorTemperatureAttr(False)
        
        print(f"Created overhead spotlight at {light_path}")
        return light_path
    except Exception as e:
        print(f"Warning: Could not create spotlight: {e}")
        return None


def update_spotlight(light_path: str, x: float, y: float, height: float):
    """Update spotlight position to follow the camera."""
    try:
        import omni.usd
        from pxr import Gf, UsdGeom
        
        stage = omni.usd.get_context().get_stage()
        light_prim = stage.GetPrimAtPath(light_path)
        
        if light_prim and light_prim.IsValid():
            xformable = UsdGeom.Xformable(light_prim)
            
            # Clear and set transform
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(x, y, height))
            
    except Exception as e:
        pass  # Silently ignore spotlight update errors


def distribute_robots_across_terrain(isaac_env, num_rows: int, num_cols: int):
    """Manually set terrain indices to distribute robots evenly across all cells."""
    num_cells = num_rows * num_cols
    num_envs = isaac_env.num_envs
    
    terrain = isaac_env.scene.terrain
    if terrain is None:
        print("Warning: No terrain found")
        return False
    
    device = isaac_env.device
    
    # Create indices for each cell (row-major order)
    terrain_levels = torch.zeros(num_envs, dtype=torch.long, device=device)
    terrain_types = torch.zeros(num_envs, dtype=torch.long, device=device)
    
    for env_idx in range(min(num_envs, num_cells)):
        row = env_idx // num_cols
        col = env_idx % num_cols
        terrain_levels[env_idx] = row
        terrain_types[env_idx] = col
    
    # Handle extra environments if any
    for env_idx in range(num_cells, num_envs):
        row = env_idx % num_rows
        col = env_idx % num_cols
        terrain_levels[env_idx] = row
        terrain_types[env_idx] = col
    
    # Assign terrain levels and types
    terrain.terrain_levels = terrain_levels
    terrain.terrain_types = terrain_types
    
    # Recompute env_origins based on terrain assignments
    if hasattr(terrain, 'terrain_origins'):
        for env_idx in range(num_envs):
            row = terrain_levels[env_idx].item()
            col = terrain_types[env_idx].item()
            if row < terrain.terrain_origins.shape[0] and col < terrain.terrain_origins.shape[1]:
                terrain.env_origins[env_idx] = terrain.terrain_origins[row, col]
    
    print(f"Distributed {num_envs} robots across {num_rows}x{num_cols} terrain grid")
    return True


def evaluate():
    """Evaluate trained policy."""
    
    # Load saved configurations
    log_dir = f"logs/{EXPERIMENT_NAME}"
    if not os.path.exists(log_dir):
        raise ValueError(f"Log directory {log_dir} not found")
    
    # Find checkpoint - use specified or find latest
    checkpoint_path = f"{log_dir}/{CHECKPOINT}"
    if not os.path.exists(checkpoint_path):
        checkpoints = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.replace("model_", "").replace(".pt", "")))
            checkpoint_path = f"{log_dir}/{checkpoints[-1]}"
            print(f"Specified checkpoint not found, using latest: {checkpoints[-1]}")
        else:
            raise ValueError(f"No checkpoints found in {log_dir}. Run training first.")
    
    # Create environment with one robot per subterrain cell
    env_cfg = QuadrupedEnvCfg()
    
    # Get terrain grid dimensions
    terrain_cfg = env_cfg.scene.terrain.terrain_generator
    num_rows = terrain_cfg.num_rows
    num_cols = terrain_cfg.num_cols
    num_cells = num_rows * num_cols
    
    # Set num_envs to match terrain grid (one robot per cell)
    env_cfg.scene.num_envs = num_cells
    
    # Disable curriculum - use random difficulty for each subterrain
    terrain_cfg.curriculum = False
    
    # Allow spawning across all terrain levels
    env_cfg.scene.terrain.max_init_terrain_level = num_rows - 1
    
    print("\n" + "=" * 80)
    print("QUADRUPED POLICY EVALUATION")
    print("=" * 80)
    print(f"Log directory: {log_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Terrain grid: {num_rows} rows x {num_cols} cols = {num_cells} cells")
    print(f"Terrain mode: Random (non-curriculum)")
    print(f"Num environments: {num_cells} (one per subterrain)")
    print(f"Camera pan duration: {CAMERA_PAN_DURATION}s")
    print(f"Headless: {HEADLESS}")
    print(f"Record video: {RECORD_VIDEO}")
    print("=" * 80 + "\n")
    
    # Create environment
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Distribute robots evenly across terrain BEFORE wrapping
    distribute_robots_across_terrain(isaac_env, num_rows, num_cols)
    
    # Wrap for RSL-RL
    env = IsaacLabVecEnvWrapper(isaac_env)
    
    # Setup frame capture
    viewport = None
    if RECORD_VIDEO or not HEADLESS:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        os.makedirs(FRAMES_DIR, exist_ok=True)
        viewport = setup_frame_capture()
        if viewport is not None:
            print(f"Frame capture enabled, saving to {FRAMES_DIR}")
    
    # Setup camera pan
    pan_config = setup_camera_pan(env, CAMERA_PAN_DURATION, CAMERA_HEIGHT, CAMERA_LOOK_AHEAD, CAMERA_TARGET_HEIGHT)
    print(f"Camera pan: {pan_config['start_pos']} -> {pan_config['end_pos']}")
    
    # Create overhead spotlight
    spotlight_path = create_spotlight(isaac_env)
    
    # Initialize camera to starting position and let it settle
    print("Initializing camera position...")
    update_camera(isaac_env, pan_config, 0.0)
    if spotlight_path:
        update_spotlight(spotlight_path, pan_config["start_pos"][0], pan_config["start_pos"][1], SPOTLIGHT_HEIGHT)
    for _ in range(60):
        simulation_app.update()
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=env.device, weights_only=False)
    
    # Get initial observation for policy construction
    obs_dict = env.reset()
    
    # Create policy with same architecture as training
    try:
        cfgs = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
        train_cfg = cfgs.get('train_cfg', None)
        if train_cfg is not None:
            policy_cfg = train_cfg.policy
        else:
            policy_cfg = {
                "class_name": "ActorCritic",
                "activation": "elu",
                "actor_hidden_dims": [256, 128, 64],
                "critic_hidden_dims": [256, 128, 64],
                "init_noise_std": 0.5,
            }
    except Exception:
        policy_cfg = {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "init_noise_std": 0.5,
        }
    
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }
    
    policy = ActorCritic(
        obs=obs_dict,
        obs_groups=obs_groups,
        num_actions=env.num_actions,
        **policy_cfg
    ).to(env.device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        policy.load_state_dict(checkpoint['model'])
    else:
        raise KeyError(f"Checkpoint missing model weights. Keys: {list(checkpoint.keys())}")
    
    policy.eval()
    print("Policy loaded successfully\n")

    # Evaluation loop - run until camera pan completes
    print("Running evaluation with camera pan...")
    if not HEADLESS:
        print("Live visualization enabled - watch the simulator window")
    print("Press Ctrl+C to stop\n")
    
    obs_dict = env.reset()
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)
    completed_episodes = 0
    step_count = 0
    frame_count = 0
    
    # Calculate total steps for camera pan
    sim_dt = env_cfg.sim.dt
    total_pan_steps = int(CAMERA_PAN_DURATION / sim_dt)
    
    # Frame skip to achieve target FPS
    sim_fps = 1.0 / sim_dt
    frame_skip = max(1, int(sim_fps / VIDEO_FPS))
    
    all_episode_rewards = []
    all_episode_lengths = []
    
    # Track termination reasons
    termination_counts = {
        "time_out": 0,
        "bad_orientation": 0,
        "base_height_low": 0,
        "illegal_contact": 0,
        "unknown": 0,
    }
    
    try:
        while step_count < total_pan_steps:
            with torch.no_grad():
                actions = policy.act_inference(obs_dict)
            
            obs_dict, rewards, dones, extras = env.step(actions)
            
            # Update camera position every step
            pan_progress = step_count / total_pan_steps
            update_camera(isaac_env, pan_config, pan_progress)
            
            # Update spotlight to follow camera
            if spotlight_path:
                start = pan_config["start_pos"]
                end = pan_config["end_pos"]
                light_x = start[0] + (end[0] - start[0]) * pan_progress
                light_y = start[1] + (end[1] - start[1]) * pan_progress
                update_spotlight(spotlight_path, light_x, light_y, SPOTLIGHT_HEIGHT)
            
            # Capture frame for video
            if RECORD_VIDEO and viewport is not None:
                if step_count % frame_skip == 0:
                    frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_count:06d}.png")
                    if capture_frame(viewport, frame_path):
                        frame_count += 1
            
            episode_rewards += rewards
            episode_lengths += 1
            step_count += 1
            
            # Print progress every 10%
            if step_count % (total_pan_steps // 10) == 0:
                progress_pct = 100.0 * step_count / total_pan_steps
                print(f"  Progress: {progress_pct:.0f}% ({step_count}/{total_pan_steps} steps)")
            
            # Handle completed episodes
            if dones.any():
                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_indices.dim() == 0:
                    done_indices = done_indices.unsqueeze(0)
                
                isaac_env_inner = env.unwrapped
                robot = isaac_env_inner.scene["robot"]
                
                for idx in done_indices:
                    idx_item = idx.item()
                    
                    height = robot.data.root_pos_w[idx_item, 2].item()
                    quat = robot.data.root_quat_w[idx_item]
                    w = quat[0].item()
                    
                    terrain_height = 0.0
                    if hasattr(isaac_env_inner.scene, 'terrain') and isaac_env_inner.scene.terrain is not None:
                        terrain = isaac_env_inner.scene.terrain
                        if hasattr(terrain, 'env_origins'):
                            terrain_height = terrain.env_origins[idx_item, 2].item()
                    height_above_terrain = height - terrain_height
                    
                    tilt_angle = 2.0 * math.acos(min(abs(w), 1.0))
                    tilt_deg = math.degrees(tilt_angle)
                    
                    reason = "unknown"
                    if episode_lengths[idx_item].item() >= env.max_episode_length:
                        reason = "time_out"
                    elif height_above_terrain < 0.1:
                        reason = "base_height_low"
                    elif tilt_deg > 45.0:
                        reason = "bad_orientation"
                    
                    termination_counts[reason] += 1
                
                for idx in done_indices:
                    completed_episodes += 1
                    ep_reward = episode_rewards[idx].item()
                    ep_length = episode_lengths[idx].item()
                    
                    all_episode_rewards.append(ep_reward)
                    all_episode_lengths.append(ep_length)
                
                episode_rewards[done_indices] = 0
                episode_lengths[done_indices] = 0
    
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    
    # Print termination summary
    print("\n" + "=" * 60)
    print("TERMINATION REASONS SUMMARY")
    print("=" * 60)
    total = sum(termination_counts.values())
    for reason, count in termination_counts.items():
        if count > 0:
            pct = 100.0 * count / total if total > 0 else 0
            print(f"  {reason}: {count} ({pct:.1f}%)")
    print("=" * 60)
    
    # Create video from frames
    if RECORD_VIDEO and frame_count > 0:
        print(f"\nCaptured {frame_count} frames")
        video_path = os.path.join(VIDEO_DIR, "evaluation.mp4")
        if frames_to_video(FRAMES_DIR, video_path, VIDEO_FPS):
            print(f"Cleaning up frames directory...")
            shutil.rmtree(FRAMES_DIR)
            os.makedirs(FRAMES_DIR, exist_ok=True)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total simulation time: {step_count * sim_dt:.1f}s")
    print(f"Total steps: {step_count}")
    if len(all_episode_rewards) > 0:
        print(f"Episodes completed: {len(all_episode_rewards)}")
        print(f"Mean reward: {np.mean(all_episode_rewards):.2f} +/- {np.std(all_episode_rewards):.2f}")
        print(f"Mean length: {np.mean(all_episode_lengths):.0f} +/- {np.std(all_episode_lengths):.0f}")
        print(f"Max reward: {np.max(all_episode_rewards):.2f}")
        print(f"Min reward: {np.min(all_episode_rewards):.2f}")
    if RECORD_VIDEO and frame_count > 0:
        print(f"Video saved to: {VIDEO_DIR}/evaluation.mp4")
    print("=" * 80)
    
    print("\nEvaluation complete!")
    
    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    evaluate()
