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
NUM_ENVS = 16
NUM_EPISODES = 32
EXPERIMENT_NAME = "quadruped_walking"
CHECKPOINT = "model_199.pt"

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

from configs import QuadrupedEnvCfg
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


def evaluate():
    """Evaluate trained policy."""
    
    # Load saved configurations
    log_dir = f"logs/{EXPERIMENT_NAME}"
    if not os.path.exists(log_dir):
        raise ValueError(f"Log directory {log_dir} not found")
    
    checkpoint_path = f"{log_dir}/{CHECKPOINT}"
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint {checkpoint_path} not found")
    
    # Create environment with fewer envs for evaluation
    env_cfg = QuadrupedEnvCfg()
    env_cfg.scene.num_envs = NUM_ENVS
    
    print("\n" + "=" * 80)
    print("QUADRUPED POLICY EVALUATION")
    print("=" * 80)
    print(f"Log directory: {log_dir}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Num environments: {NUM_ENVS}")
    print(f"Num episodes: {NUM_EPISODES}")
    print(f"Headless: {HEADLESS}")
    print(f"Record video: {RECORD_VIDEO}")
    print("=" * 80 + "\n")
    
    # Create environment
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = IsaacLabVecEnvWrapper(isaac_env)
    
    # Setup frame capture
    viewport = None
    if RECORD_VIDEO:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        os.makedirs(FRAMES_DIR, exist_ok=True)
        viewport = setup_frame_capture()
        if viewport is not None:
            print(f"Frame capture enabled, saving to {FRAMES_DIR}")
    
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
    
    # Track termination reasons
    termination_counts = {
        "time_out": 0,
        "bad_orientation": 0,
        "base_height_low": 0,
        "illegal_contact": 0,
        "total_resets": 0,
    }

    # Evaluation loop
    print("Running evaluation...")
    if not HEADLESS:
        print("Live visualization enabled - watch the simulator window")
    print("Press Ctrl+C to stop\n")
    
    obs_dict = env.reset()
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)
    completed_episodes = 0
    step_count = 0
    frame_count = 0
    episode_start_frame = 0
    
    # Frame skip to achieve target FPS
    sim_fps = 1.0 / env_cfg.sim.dt
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
        while completed_episodes < NUM_EPISODES:
            with torch.no_grad():
                actions = policy.act_inference(obs_dict)
            
            obs_dict, rewards, dones, extras = env.step(actions)
            
            # Capture frame for video
            if RECORD_VIDEO and viewport is not None:
                if step_count % frame_skip == 0:
                    frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_count:06d}.png")
                    if capture_frame(viewport, frame_path):
                        frame_count += 1
            
            episode_rewards += rewards
            episode_lengths += 1
            step_count += 1
            
            # Debug termination reasons when episodes end
            if dones.any():
                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_indices.dim() == 0:
                    done_indices = done_indices.unsqueeze(0)
                
                # Get termination info from Isaac Lab environment
                isaac_env = env.unwrapped
                robot = isaac_env.scene["robot"]
                
                # Get body names for contact sensor
                contact_sensor = isaac_env.scene.sensors["contact_forces"]
                body_names = []
                if hasattr(contact_sensor, 'body_names'):
                    body_names = contact_sensor.body_names
                elif hasattr(contact_sensor, 'cfg') and hasattr(contact_sensor.cfg, 'prim_path'):
                    # Try to extract from prim path pattern
                    body_names = [f"body_{i}" for i in range(contact_sensor.data.net_forces_w.shape[1])]
                
                for idx in done_indices:
                    idx_item = idx.item()
                    
                    # Check each termination condition manually
                    height = robot.data.root_pos_w[idx_item, 2].item()
                    quat = robot.data.root_quat_w[idx_item]
                    w = quat[0].item()
                    
                    # Get terrain height for relative height calculation
                    terrain_height = 0.0
                    if hasattr(isaac_env.scene, 'terrain') and isaac_env.scene.terrain is not None:
                        terrain = isaac_env.scene.terrain
                        if hasattr(terrain, 'env_origins'):
                            terrain_height = terrain.env_origins[idx_item, 2].item()
                    height_above_terrain = height - terrain_height
                    
                    # Compute tilt angle from quaternion w component
                    tilt_angle = 2.0 * math.acos(min(abs(w), 1.0))
                    tilt_deg = math.degrees(tilt_angle)
                    
                    # Check contact forces and identify which bodies
                    forces = contact_sensor.data.net_forces_w[idx_item]
                    force_mags = torch.norm(forces, dim=-1)
                    max_force = force_mags.max().item()
                    
                    # Find bodies with contact force above threshold
                    contact_threshold = 10.0
                    contacting_bodies = []
                    for body_idx, force_mag in enumerate(force_mags):
                        if force_mag.item() > contact_threshold:
                            body_name = body_names[body_idx] if body_idx < len(body_names) else f"body_{body_idx}"
                            contacting_bodies.append((body_name, force_mag.item()))
                    
                    # Determine termination reason
                    reason = "unknown"
                    if episode_lengths[idx_item].item() >= env.max_episode_length:
                        reason = "time_out"
                    elif height_above_terrain < 0.1:
                        reason = "base_height_low"
                    elif tilt_deg > 45.0:
                        reason = "bad_orientation"
                    elif max_force > contact_threshold:
                        reason = "illegal_contact"
                    
                    termination_counts[reason] += 1
                    
                    # Print debug info for all non-timeout terminations
                    if reason != "time_out":
                        print(f"\n  TERMINATION DEBUG (Episode {completed_episodes + 1}):")
                        print(f"    Reason: {reason}")
                        print(f"    Height above terrain: {height_above_terrain:.3f}m (threshold: 0.1m)")
                        print(f"    World height: {height:.3f}m, Terrain height: {terrain_height:.3f}m")
                        print(f"    Tilt: {tilt_deg:.1f} deg (threshold: 45 deg)")
                        print(f"    Max contact force: {max_force:.1f}N (threshold: {contact_threshold}N)")
                        if contacting_bodies:
                            print(f"    Bodies in contact:")
                            for body_name, force in contacting_bodies:
                                print(f"      - {body_name}: {force:.1f}N")
                        print(f"    Episode length: {episode_lengths[idx_item].item():.0f} steps")
                
                # Record completed episodes
                for idx in done_indices:
                    completed_episodes += 1
                    ep_reward = episode_rewards[idx].item()
                    ep_length = episode_lengths[idx].item()
                    
                    all_episode_rewards.append(ep_reward)
                    all_episode_lengths.append(ep_length)
                    
                    print(f"Episode {completed_episodes}: "
                          f"Reward = {ep_reward:.2f}, "
                          f"Length = {ep_length:.0f}, "
                          f"Frames = {frame_count - episode_start_frame}")
                    
                    episode_start_frame = frame_count
                    
                    if completed_episodes >= NUM_EPISODES:
                        break
                
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
            # Clean up frames directory
            print(f"Cleaning up frames directory...")
            shutil.rmtree(FRAMES_DIR)
            os.makedirs(FRAMES_DIR, exist_ok=True)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
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
