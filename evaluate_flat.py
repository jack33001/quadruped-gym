"""
Evaluation script for trained quadruped policies on flat ground.
Loads checkpoint and runs evaluation with optional video recording.
"""
import os
import sys

os.environ["PYTORCH_NVFUSER_DISABLE_FALLBACK"] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

# Configuration - edit these values as needed
HEADLESS = False
RECORD_VIDEO = True
VIDEO_DIR = "logs/quadruped_walking/videos_flat"
FRAMES_DIR = "logs/quadruped_walking/frames_flat"
VIDEO_FPS = 30
NUM_ENVS = 16
EXPERIMENT_NAME = "quadruped_walking"
CHECKPOINT = "model_499.pt"

# Camera configuration
CAMERA_HEIGHT = 3.0
CAMERA_DISTANCE = 5.0
EVALUATION_DURATION = 20.0

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

import pickle
import torch
import numpy as np
import shutil
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from rsl_rl.modules import ActorCritic

from rl_cfg import QuadrupedEnvCfg
from sim_cfg import QUADRUPED_CFG
from quadruped_env import IsaacLabVecEnvWrapper


@configclass
class FlatGroundSceneCfg(InteractiveSceneCfg):
    """Scene configuration with flat ground plane instead of terrain."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot = QUADRUPED_CFG

    from isaaclab.sensors import ContactSensorCfg, ImuCfg

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Quadruped/Torso/Torso",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )

    torso_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Quadruped/Torso/Torso",
        history_length=3,
        track_air_time=False,
        update_period=0.0,
    )

    thigh_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Quadruped/Torso/.*_Thigh",
        history_length=3,
        track_air_time=False,
        update_period=0.0,
    )

    shin_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Quadruped/Torso/.*_Shin",
        history_length=3,
        track_air_time=False,
        update_period=0.0,
    )

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

        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        if len(frame_files) == 0:
            print(f"No frames found in {frames_dir}")
            return False

        print(f"Combining {len(frame_files)} frames into video...")

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


def set_camera_view(isaac_env, eye: tuple, target: tuple):
    """Set camera position and look-at target."""
    try:
        import numpy as np
        cam_eye = np.array(eye, dtype=float)
        cam_target = np.array(target, dtype=float)
        isaac_env.sim.set_camera_view(eye=cam_eye, target=cam_target)
        return True
    except Exception as e:
        return False


def evaluate():
    """Evaluate trained policy on flat ground."""

    log_dir = f"logs/{EXPERIMENT_NAME}"
    if not os.path.exists(log_dir):
        raise ValueError(f"Log directory {log_dir} not found")

    checkpoint_path = f"{log_dir}/{CHECKPOINT}"
    if not os.path.exists(checkpoint_path):
        checkpoints = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.replace("model_", "").replace(".pt", "")))
            checkpoint_path = f"{log_dir}/{checkpoints[-1]}"
            print(f"Specified checkpoint not found, using latest: {checkpoints[-1]}")
        else:
            raise ValueError(f"No checkpoints found in {log_dir}. Run training first.")

    env_cfg = QuadrupedEnvCfg()
    env_cfg.scene = FlatGroundSceneCfg(num_envs=NUM_ENVS, env_spacing=2.5)

    print("\n" + "=" * 80)
    print("QUADRUPED POLICY EVALUATION (FLAT GROUND)")
    print("=" * 80)
    print(f"Log directory: {log_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Num environments: {NUM_ENVS}")
    print(f"Evaluation duration: {EVALUATION_DURATION}s")
    print(f"Headless: {HEADLESS}")
    print(f"Record video: {RECORD_VIDEO}")
    print("=" * 80 + "\n")

    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = IsaacLabVecEnvWrapper(isaac_env)

    viewport = None
    if RECORD_VIDEO or not HEADLESS:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        os.makedirs(FRAMES_DIR, exist_ok=True)
        viewport = setup_frame_capture()
        if viewport is not None:
            print(f"Frame capture enabled, saving to {FRAMES_DIR}")

    set_camera_view(isaac_env, 
                    eye=(CAMERA_DISTANCE, CAMERA_DISTANCE, CAMERA_HEIGHT),
                    target=(0.0, 0.0, 0.0))
    for _ in range(60):
        simulation_app.update()

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=env.device, weights_only=False)

    obs_dict = env.reset()

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

    if 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        policy.load_state_dict(checkpoint['model'])
    else:
        raise KeyError(f"Checkpoint missing model weights. Keys: {list(checkpoint.keys())}")

    policy.eval()
    print("Policy loaded successfully\n")

    print("Running evaluation on flat ground...")
    if not HEADLESS:
        print("Live visualization enabled - watch the simulator window")
    print("Press Ctrl+C to stop\n")

    obs_dict = env.reset()
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)
    step_count = 0
    frame_count = 0

    sim_dt = env_cfg.sim.dt
    total_steps = int(EVALUATION_DURATION / sim_dt)

    sim_fps = 1.0 / sim_dt
    frame_skip = max(1, int(sim_fps / VIDEO_FPS))

    all_episode_rewards = []
    all_episode_lengths = []

    termination_counts = {
        "time_out": 0,
        "bad_orientation": 0,
        "base_height_low": 0,
        "illegal_contact": 0,
        "unknown": 0,
    }

    try:
        while step_count < total_steps:
            with torch.no_grad():
                actions = policy.act_inference(obs_dict)

            obs_dict, rewards, dones, extras = env.step(actions)

            if RECORD_VIDEO and viewport is not None:
                if step_count % frame_skip == 0:
                    frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_count:06d}.png")
                    if capture_frame(viewport, frame_path):
                        frame_count += 1

            episode_rewards += rewards
            episode_lengths += 1
            step_count += 1

            if step_count % (total_steps // 10) == 0:
                progress_pct = 100.0 * step_count / total_steps
                print(f"  Progress: {progress_pct:.0f}% ({step_count}/{total_steps} steps)")

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

                    tilt_angle = 2.0 * math.acos(min(abs(w), 1.0))
                    tilt_deg = math.degrees(tilt_angle)

                    reason = "unknown"
                    if episode_lengths[idx_item].item() >= env.max_episode_length:
                        reason = "time_out"
                    elif height < 0.1:
                        reason = "base_height_low"
                    elif tilt_deg > 45.0:
                        reason = "bad_orientation"

                    termination_counts[reason] += 1

                for idx in done_indices:
                    ep_reward = episode_rewards[idx].item()
                    ep_length = episode_lengths[idx].item()

                    all_episode_rewards.append(ep_reward)
                    all_episode_lengths.append(ep_length)

                episode_rewards[done_indices] = 0
                episode_lengths[done_indices] = 0

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")

    print("\n" + "=" * 60)
    print("TERMINATION REASONS SUMMARY")
    print("=" * 60)
    total = sum(termination_counts.values())
    for reason, count in termination_counts.items():
        if count > 0:
            pct = 100.0 * count / total if total > 0 else 0
            print(f"  {reason}: {count} ({pct:.1f}%)")
    print("=" * 60)

    if RECORD_VIDEO and frame_count > 0:
        print(f"\nCaptured {frame_count} frames")
        video_path = os.path.join(VIDEO_DIR, "evaluation_flat.mp4")
        if frames_to_video(FRAMES_DIR, video_path, VIDEO_FPS):
            print(f"Cleaning up frames directory...")
            shutil.rmtree(FRAMES_DIR)
            os.makedirs(FRAMES_DIR, exist_ok=True)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY (FLAT GROUND)")
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
        print(f"Video saved to: {VIDEO_DIR}/evaluation_flat.mp4")
    print("=" * 80)

    print("\nEvaluation complete!")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    evaluate()
