"""
Gait performance evaluation script.

Tests each gait at multiple velocities with video recording and overlays.
"""
import os

os.environ["PYTORCH_NVFUSER_DISABLE_FALLBACK"] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

from isaaclab.app import AppLauncher

from train_cfg import TrainCfg

TRAIN_CFG = TrainCfg()
EVAL_CFG = TRAIN_CFG.eval

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from isaaclab.envs import ManagerBasedRLEnv

from rl_cfg import QuadrupedEnvCfg
from sim_cfg import FlatGroundSceneCfg
from quadruped_env import IsaacLabVecEnvWrapper
from evaluate_base import find_checkpoint, load_policy, set_camera_view, capture_frame, setup_frame_capture
from gait_cfg import GaitType, GAIT_PARAMS, GaitSchedulerCfg


class GaitEvaluator:
    """Evaluates gait performance across velocities with video recording."""
    
    def __init__(self, train_cfg: TrainCfg):
        self.train_cfg = train_cfg
        self.log_dir = f"logs/{train_cfg.experiment_name}"
        self.output_dir = f"{self.log_dir}/gait_eval"
        self.frames_dir = f"{self.output_dir}/frames"
        
        self.velocity_range = (0.0, 2.0)
        self.velocity_step = 0.5
        self.test_duration = 5.0
        self.video_fps = 30
        
        self.camera_distance = 1.5
        self.camera_height = 0.8
        self.camera_side_offset = 1.5
        
        self.viewport = None
        
        gait_cfg = GaitSchedulerCfg()
        self.enabled_gaits = list(gait_cfg.enabled_gaits)
    
    def get_test_velocities(self) -> list:
        """Generate list of velocities to test."""
        velocities = []
        v = self.velocity_range[0]
        while v <= self.velocity_range[1] + 0.01:
            if v > 0.01:
                velocities.append(round(v, 1))
            v += self.velocity_step
        return velocities
    
    def setup_dirs(self):
        """Create output directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
    
    def create_env(self) -> tuple:
        """Create environment with single robot."""
        env_cfg = QuadrupedEnvCfg()
        env_cfg.scene = FlatGroundSceneCfg(num_envs=1, env_spacing=2.5)
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        
        isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
        env = IsaacLabVecEnvWrapper(isaac_env)
        
        return env, isaac_env, env_cfg
    
    def set_velocity_command(self, isaac_env, velocity: float):
        """Set the velocity command for the environment."""
        cmd_manager = isaac_env.command_manager
        if hasattr(cmd_manager, 'get_command'):
            velocity_cmd = cmd_manager.get_command("base_velocity")
            if velocity_cmd is not None:
                velocity_cmd[:, 0] = velocity
                velocity_cmd[:, 1] = 0.0
                velocity_cmd[:, 2] = 0.0
    
    def set_gait(self, env, gait_type: GaitType):
        """Set the gait type for the environment."""
        env.gait_scheduler.gait_types[0] = gait_type
        env.gait_scheduler.cfg.allow_gait_switching = False
    
    def update_camera(self, isaac_env, robot_pos: np.ndarray):
        """Update camera to follow robot from the side."""
        eye = (
            robot_pos[0] - self.camera_distance,
            robot_pos[1] + self.camera_side_offset,
            self.camera_height
        )
        target = (robot_pos[0], robot_pos[1], 0.3)
        set_camera_view(isaac_env, eye=eye, target=target)
    
    def capture_frame(self) -> np.ndarray:
        """Capture frame from viewport."""
        try:
            from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_buffer
            viewport = get_active_viewport()
            if viewport is None:
                return None
            
            buffer = capture_viewport_to_buffer(viewport)
            if buffer is not None:
                return np.array(buffer)
        except Exception:
            pass
        return None
    
    def add_overlay(self, frame: np.ndarray, gait_name: str, velocity: float, 
                    avg_error: float, actual_vel: float) -> np.ndarray:
        """Add text overlay to frame."""
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
            font_small = font
        
        padding = 10
        line_height = 28
        
        lines = [
            f"Gait: {gait_name.upper()}",
            f"Commanded: {velocity:.1f} m/s",
            f"Actual: {actual_vel:.2f} m/s",
            f"Error: {avg_error:.1f}%",
        ]
        
        bg_height = len(lines) * line_height + padding * 2
        bg_width = 250
        
        overlay = Image.new('RGBA', (bg_width, bg_height), (0, 0, 0, 180))
        img.paste(overlay, (padding, padding), overlay)
        
        y = padding + 5
        for i, line in enumerate(lines):
            f = font if i == 0 else font_small
            draw.text((padding + 10, y), line, font=f, fill=(255, 255, 255))
            y += line_height
        
        return np.array(img)
    
    def run_single_test(self, env, isaac_env, policy, gait_type: GaitType, 
                        velocity: float, sim_dt: float) -> tuple:
        """Run a single gait/velocity test and capture frames."""
        gait_name = GAIT_PARAMS[gait_type].name
        print(f"  Testing {gait_name} at {velocity:.1f} m/s...")
        
        self.set_gait(env, gait_type)
        self.set_velocity_command(isaac_env, velocity)
        
        obs_dict = env.reset()
        self.set_gait(env, gait_type)
        self.set_velocity_command(isaac_env, velocity)
        
        total_steps = int(self.test_duration / sim_dt)
        frame_skip = max(1, int((1.0 / sim_dt) / self.video_fps))
        
        frame_idx = 0
        velocity_errors = []
        actual_velocities = []
        
        test_frames_dir = os.path.join(self.frames_dir, f"{gait_name}_{velocity:.1f}")
        os.makedirs(test_frames_dir, exist_ok=True)
        
        for step in range(total_steps):
            with torch.no_grad():
                actions = policy.act_inference(obs_dict)
            
            obs_dict, _, dones, _ = env.step(actions)
            
            self.set_velocity_command(isaac_env, velocity)
            
            robot = isaac_env.scene["robot"]
            robot_pos = robot.data.root_pos_w[0].cpu().numpy()
            actual_vel = robot.data.root_lin_vel_w[0, 0].item()
            
            actual_velocities.append(actual_vel)
            if velocity > 0.01:
                error = abs(actual_vel - velocity) / velocity * 100
                velocity_errors.append(error)
            
            self.update_camera(isaac_env, robot_pos)
            
            if step % frame_skip == 0:
                simulation_app.update()
                
                if self.viewport is not None:
                    frame_path = os.path.join(test_frames_dir, f"frame_{frame_idx:06d}.png")
                    if capture_frame(self.viewport, frame_path):
                        frame_idx += 1
            
            if dones.any():
                obs_dict = env.reset()
                self.set_gait(env, gait_type)
                self.set_velocity_command(isaac_env, velocity)
        
        avg_error = np.mean(velocity_errors) if velocity_errors else 0.0
        avg_actual = np.mean(actual_velocities) if actual_velocities else 0.0
        
        return test_frames_dir, frame_idx, avg_error, avg_actual, gait_name
    
    def save_video_with_overlay(self, frames_dir: str, output_path: str, 
                                 gait_name: str, velocity: float, 
                                 avg_error: float, avg_actual: float) -> bool:
        """Save frames as video with text overlay using ffmpeg."""
        import subprocess
        import shutil
        
        frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
        
        if not os.path.exists(os.path.join(frames_dir, "frame_000000.png")):
            print(f"No frames found in {frames_dir}")
            return False
        
        drawtext_filter = (
            f"drawtext=text='Gait\\: {gait_name.upper()}':"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
            f"fontsize=24:fontcolor=white:x=20:y=20:"
            f"box=1:boxcolor=black@0.7:boxborderw=5,"
            f"drawtext=text='Commanded\\: {velocity:.1f} m/s':"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
            f"fontsize=18:fontcolor=white:x=20:y=55:"
            f"box=1:boxcolor=black@0.7:boxborderw=5,"
            f"drawtext=text='Actual\\: {avg_actual:.2f} m/s':"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
            f"fontsize=18:fontcolor=white:x=20:y=80:"
            f"box=1:boxcolor=black@0.7:boxborderw=5,"
            f"drawtext=text='Error\\: {avg_error:.1f}%':"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
            f"fontsize=18:fontcolor=white:x=20:y=105:"
            f"box=1:boxcolor=black@0.7:boxborderw=5"
        )
        
        try:
            result = subprocess.run([
                'ffmpeg', '-y',
                '-framerate', str(self.video_fps),
                '-i', frame_pattern,
                '-vf', drawtext_filter,
                '-c:v', 'h264_nvenc', '-preset', 'fast', '-b:v', '5M',
                output_path
            ], capture_output=True, timeout=120)
            
            if result.returncode == 0:
                shutil.rmtree(frames_dir)
                return True
            
            result = subprocess.run([
                'ffmpeg', '-y',
                '-framerate', str(self.video_fps),
                '-i', frame_pattern,
                '-vf', drawtext_filter,
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                output_path
            ], capture_output=True, timeout=120)
            
            if result.returncode == 0:
                shutil.rmtree(frames_dir)
                return True
            else:
                print(f"ffmpeg error: {result.stderr.decode()}")
                return False
                
        except Exception as e:
            print(f"Error encoding video: {e}")
            return False
    
    def combine_videos(self, video_paths: list, output_path: str):
        """Combine multiple videos into one using ffmpeg."""
        import subprocess
        
        if not video_paths:
            return False
        
        existing_paths = [p for p in video_paths if os.path.exists(p)]
        if not existing_paths:
            return False
            
        list_file = os.path.join(self.output_dir, 'videos.txt')
        with open(list_file, 'w') as f:
            for path in existing_paths:
                f.write(f"file '{os.path.abspath(path)}'\n")
        
        try:
            result = subprocess.run([
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', list_file,
                '-c:v', 'h264_nvenc', '-preset', 'fast', '-b:v', '5M',
                output_path
            ], capture_output=True, timeout=180)
            
            if result.returncode != 0:
                result = subprocess.run([
                    'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                    '-i', list_file,
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    output_path
                ], capture_output=True, timeout=180)
            
            os.remove(list_file)
            
            if result.returncode == 0:
                print(f"Combined video saved to: {output_path}")
                return True
            else:
                print(f"ffmpeg concat error: {result.stderr.decode()}")
                return False
                
        except Exception as e:
            print(f"Error combining videos: {e}")
            if os.path.exists(list_file):
                os.remove(list_file)
            return False
    
    def run(self):
        """Run the full gait evaluation."""
        self.setup_dirs()
        
        if not os.path.exists(self.log_dir):
            raise ValueError(f"Log directory {self.log_dir} not found")
        
        checkpoint_path = find_checkpoint(self.log_dir, f"model_{self.train_cfg.max_iterations - 1}.pt")
        
        print("\n" + "=" * 80)
        print("GAIT PERFORMANCE EVALUATION")
        print("=" * 80)
        print(f"Log directory: {self.log_dir}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Velocities: {self.get_test_velocities()}")
        print(f"Gaits: {[GAIT_PARAMS[gt].name for gt in self.enabled_gaits]}")
        print(f"Test duration per combo: {self.test_duration}s")
        print("=" * 80 + "\n")
        
        env, isaac_env, env_cfg = self.create_env()
        sim_dt = env_cfg.sim.dt
        
        self.viewport = setup_frame_capture()
        if self.viewport is None:
            print("Warning: Frame capture not available, videos will not be recorded")
        
        for _ in range(60):
            simulation_app.update()
        
        obs_dict = env.reset()
        policy = load_policy(self.log_dir, checkpoint_path, obs_dict, env.num_actions, env.device)
        print("Policy loaded successfully\n")
        
        velocities = self.get_test_velocities()
        results = {}
        video_paths = []
        
        for gait_type in self.enabled_gaits:
            gait_name = GAIT_PARAMS[gait_type].name
            print(f"\nTesting gait: {gait_name}")
            results[gait_name] = {}
            
            for velocity in velocities:
                frames_dir, frame_count, avg_error, avg_actual, gait_name = self.run_single_test(
                    env, isaac_env, policy, gait_type, velocity, sim_dt
                )
                
                results[gait_name][velocity] = {
                    "avg_error": avg_error,
                    "avg_actual": avg_actual,
                }
                
                video_name = f"{gait_name}_{velocity:.1f}mps.mp4"
                video_path = os.path.join(self.output_dir, video_name)
                
                if frame_count > 0 and self.save_video_with_overlay(
                    frames_dir, video_path, gait_name, velocity, avg_error, avg_actual
                ):
                    video_paths.append(video_path)
                    print(f"    Saved: {video_name} (error: {avg_error:.1f}%)")
                else:
                    print(f"    {gait_name} @ {velocity:.1f}m/s: error {avg_error:.1f}%")
        
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"{'Gait':<10} | " + " | ".join([f"{v:.1f}m/s" for v in velocities]))
        print("-" * 60)
        for gait_name, vel_results in results.items():
            errors = [f"{vel_results[v]['avg_error']:5.1f}%" for v in velocities]
            print(f"{gait_name:<10} | " + " | ".join(errors))
        print("=" * 60)
        
        combined_path = os.path.join(self.output_dir, "all_gaits_combined.mp4")
        self.combine_videos(video_paths, combined_path)
        
        print(f"\nIndividual videos saved to: {self.output_dir}")
        print(f"Combined video: {combined_path}")
        print("\nGait evaluation complete!")
        
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    evaluator = GaitEvaluator(TRAIN_CFG)
    evaluator.run()
