"""
Base evaluation utilities and abstract evaluator for quadruped policies.
"""
import os
import math
import pickle
import shutil
from abc import ABC, abstractmethod

import torch
import numpy as np

from rsl_rl.modules import ActorCritic

from rl_cfg import QuadrupedEnvCfg
from quadruped_env import IsaacLabVecEnvWrapper


def setup_frame_capture():
    """Setup frame capture using viewport.
    
    Returns:
        Viewport object if successful, None otherwise.
    """
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


def capture_frame(viewport, frame_path: str) -> bool:
    """Capture a single frame from the viewport and save to disk.
    
    Args:
        viewport: The viewport to capture from.
        frame_path: Path to save the captured frame.
        
    Returns:
        True if capture succeeded, False otherwise.
    """
    try:
        from omni.kit.viewport.utility import capture_viewport_to_file
        capture_viewport_to_file(viewport, frame_path)
        return True
    except Exception:
        return False


def frames_to_video(frames_dir: str, output_path: str, fps: int = 30) -> bool:
    """Combine saved frames into a video using imageio.
    
    Args:
        frames_dir: Directory containing frame images.
        output_path: Path for output video file.
        fps: Frames per second for output video.
        
    Returns:
        True if video creation succeeded, False otherwise.
    """
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


def set_camera_view(isaac_env, eye: tuple, target: tuple) -> bool:
    """Set camera position and look-at target.
    
    Args:
        isaac_env: The Isaac environment instance.
        eye: Camera position (x, y, z).
        target: Look-at target position (x, y, z).
        
    Returns:
        True if camera was set successfully, False otherwise.
    """
    try:
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


def find_checkpoint(log_dir: str, checkpoint_name: str) -> str:
    """Find checkpoint file, falling back to latest if specified not found.
    
    Args:
        log_dir: Directory containing checkpoints.
        checkpoint_name: Desired checkpoint filename.
        
    Returns:
        Path to checkpoint file.
        
    Raises:
        ValueError: If no checkpoints are found.
    """
    checkpoint_path = os.path.join(log_dir, checkpoint_name)
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    checkpoints = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.replace("model_", "").replace(".pt", "")))
        checkpoint_path = os.path.join(log_dir, checkpoints[-1])
        print(f"Specified checkpoint not found, using latest: {checkpoints[-1]}")
        return checkpoint_path

    raise ValueError(f"No checkpoints found in {log_dir}. Run training first.")


def load_policy(log_dir: str, checkpoint_path: str, obs_dict, num_actions: int, device) -> ActorCritic:
    """Load policy from checkpoint with architecture from saved config.
    
    Args:
        log_dir: Directory containing config files.
        checkpoint_path: Path to model checkpoint.
        obs_dict: Observation dictionary for policy construction.
        num_actions: Number of action dimensions.
        device: Torch device.
        
    Returns:
        Loaded and initialized ActorCritic policy.
        
    Raises:
        KeyError: If checkpoint is missing model weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    try:
        cfgs = pickle.load(open(os.path.join(log_dir, "cfgs.pkl"), "rb"))
        train_cfg = cfgs.get('train_cfg', None)
        if train_cfg is not None:
            policy_cfg = train_cfg.policy
        else:
            policy_cfg = _default_policy_cfg()
    except Exception:
        policy_cfg = _default_policy_cfg()

    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    policy = ActorCritic(
        obs=obs_dict,
        obs_groups=obs_groups,
        num_actions=num_actions,
        **policy_cfg
    ).to(device)

    if 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        policy.load_state_dict(checkpoint['model'])
    else:
        raise KeyError(f"Checkpoint missing model weights. Keys: {list(checkpoint.keys())}")

    policy.eval()
    return policy


def _default_policy_cfg() -> dict:
    """Default policy configuration for ActorCritic network."""
    return {
        "class_name": "ActorCritic",
        "activation": "lrelu",
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "init_noise_std": 0.5,
    }


class BaseEvaluator(ABC):
    """Abstract base class for policy evaluation."""

    def __init__(
        self,
        experiment_name: str,
        checkpoint: str,
        headless: bool = False,
        record_video: bool = True,
        video_fps: int = 30,
        evaluation_duration: float = 20.0,
    ):
        self.experiment_name = experiment_name
        self.checkpoint = checkpoint
        self.headless = headless
        self.record_video = record_video
        self.video_fps = video_fps
        self.evaluation_duration = evaluation_duration

        self.log_dir = f"logs/{experiment_name}"
        self.video_dir = f"{self.log_dir}/videos"
        self.frames_dir = f"{self.log_dir}/frames"

        self.termination_counts = {
            "time_out": 0,
            "bad_orientation": 0,
            "base_height_low": 0,
            "illegal_contact": 0,
            "unknown": 0,
        }
        
        self._frame_capture_failures = 0
        self._max_frame_failures_to_report = 5

    @abstractmethod
    def create_env_cfg(self) -> QuadrupedEnvCfg:
        """Create environment configuration."""
        pass

    @abstractmethod
    def setup_environment(self, isaac_env):
        """Setup environment-specific configuration after creation."""
        pass

    @abstractmethod
    def setup_camera(self, isaac_env):
        """Setup initial camera position."""
        pass

    def update_camera(self, isaac_env, step_count: int, total_steps: int):
        """Update camera position during evaluation. Override for dynamic camera."""
        pass

    def get_video_filename(self) -> str:
        """Get output video filename."""
        return "evaluation.mp4"

    def print_header(self, checkpoint_path: str, env_cfg: QuadrupedEnvCfg):
        """Print evaluation header information."""
        print("\n" + "=" * 80)
        print("QUADRUPED POLICY EVALUATION")
        print("=" * 80)
        print(f"Log directory: {self.log_dir}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Evaluation duration: {self.evaluation_duration}s")
        print(f"Headless: {self.headless}")
        print(f"Record video: {self.record_video}")
        print("=" * 80 + "\n")

    def classify_termination(self, env, idx: int, episode_length: float) -> str:
        """Classify termination reason for an episode.
        
        Args:
            env: The wrapped environment.
            idx: Environment index.
            episode_length: Length of the episode in steps.
            
        Returns:
            String describing the termination reason.
        """
        isaac_env = env.unwrapped
        robot = isaac_env.scene["robot"]

        height = robot.data.root_pos_w[idx, 2].item()
        quat = robot.data.root_quat_w[idx]
        w = quat[0].item()

        terrain_height = 0.0
        if hasattr(isaac_env.scene, 'terrain') and isaac_env.scene.terrain is not None:
            terrain = isaac_env.scene.terrain
            if hasattr(terrain, 'env_origins'):
                terrain_height = terrain.env_origins[idx, 2].item()
        height_above_terrain = height - terrain_height

        tilt_angle = 2.0 * math.acos(min(abs(w), 1.0))
        tilt_deg = math.degrees(tilt_angle)

        if episode_length >= env.max_episode_length:
            return "time_out"
        elif height_above_terrain < 0.1:
            return "base_height_low"
        elif tilt_deg > 45.0:
            return "bad_orientation"
        return "unknown"

    def print_termination_summary(self):
        """Print summary of termination reasons."""
        print("\n" + "=" * 60)
        print("TERMINATION REASONS SUMMARY")
        print("=" * 60)
        total = sum(self.termination_counts.values())
        for reason, count in self.termination_counts.items():
            if count > 0:
                pct = 100.0 * count / total if total > 0 else 0
                print(f"  {reason}: {count} ({pct:.1f}%)")
        print("=" * 60)

    def print_summary(self, step_count: int, sim_dt: float, all_episode_rewards: list, all_episode_lengths: list, frame_count: int):
        """Print final evaluation summary."""
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
        if self.record_video and frame_count > 0:
            print(f"Video saved to: {self.video_dir}/{self.get_video_filename()}")
        if self._frame_capture_failures > 0:
            print(f"Frame capture failures: {self._frame_capture_failures}")
        print("=" * 80)

    def run(self, simulation_app):
        """Run the evaluation loop.
        
        Args:
            simulation_app: The Isaac Sim application instance.
        """
        from isaaclab.envs import ManagerBasedRLEnv

        if not os.path.exists(self.log_dir):
            raise ValueError(f"Log directory {self.log_dir} not found")

        checkpoint_path = find_checkpoint(self.log_dir, self.checkpoint)
        env_cfg = self.create_env_cfg()
        self.print_header(checkpoint_path, env_cfg)

        isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
        self.setup_environment(isaac_env)
        env = IsaacLabVecEnvWrapper(isaac_env)

        viewport = None
        if self.record_video or not self.headless:
            os.makedirs(self.video_dir, exist_ok=True)
            os.makedirs(self.frames_dir, exist_ok=True)
            viewport = setup_frame_capture()
            if viewport is not None:
                print(f"Frame capture enabled, saving to {self.frames_dir}")

        self.setup_camera(isaac_env)
        for _ in range(60):
            simulation_app.update()

        print(f"Loading checkpoint: {checkpoint_path}")
        obs_dict = env.reset()
        policy = load_policy(self.log_dir, checkpoint_path, obs_dict, env.num_actions, env.device)
        print("Policy loaded successfully\n")

        print("Running evaluation...")
        if not self.headless:
            print("Live visualization enabled - watch the simulator window")
        print("Press Ctrl+C to stop\n")

        obs_dict = env.reset()
        episode_rewards = torch.zeros(env.num_envs, device=env.device)
        episode_lengths = torch.zeros(env.num_envs, device=env.device)
        step_count = 0
        frame_count = 0

        sim_dt = env_cfg.sim.dt
        total_steps = int(self.evaluation_duration / sim_dt)
        sim_fps = 1.0 / sim_dt
        frame_skip = max(1, int(sim_fps / self.video_fps))

        all_episode_rewards = []
        all_episode_lengths = []

        try:
            while step_count < total_steps:
                with torch.no_grad():
                    actions = policy.act_inference(obs_dict)

                obs_dict, rewards, dones, extras = env.step(actions)

                self.update_camera(isaac_env, step_count, total_steps)

                if self.record_video and viewport is not None:
                    if step_count % frame_skip == 0:
                        frame_path = os.path.join(self.frames_dir, f"frame_{frame_count:06d}.png")
                        if capture_frame(viewport, frame_path):
                            frame_count += 1
                        else:
                            self._frame_capture_failures += 1
                            if self._frame_capture_failures <= self._max_frame_failures_to_report:
                                print(f"Warning: Frame capture failed at step {step_count}")

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

                    for idx in done_indices:
                        idx_item = idx.item()
                        reason = self.classify_termination(env, idx_item, episode_lengths[idx_item].item())
                        self.termination_counts[reason] += 1

                        all_episode_rewards.append(episode_rewards[idx_item].item())
                        all_episode_lengths.append(episode_lengths[idx_item].item())

                    episode_rewards[done_indices] = 0
                    episode_lengths[done_indices] = 0

        except KeyboardInterrupt:
            print("\n\nEvaluation interrupted by user")

        self.print_termination_summary()

        if self.record_video and frame_count > 0:
            print(f"\nCaptured {frame_count} frames")
            video_path = os.path.join(self.video_dir, self.get_video_filename())
            if frames_to_video(self.frames_dir, video_path, self.video_fps):
                print("Cleaning up frames directory...")
                shutil.rmtree(self.frames_dir)
                os.makedirs(self.frames_dir, exist_ok=True)

        self.print_summary(step_count, sim_dt, all_episode_rewards, all_episode_lengths, frame_count)
        print("\nEvaluation complete!")

        env.close()
        simulation_app.close()
