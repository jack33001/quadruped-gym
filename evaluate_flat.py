"""
Evaluation script for trained quadruped policies on flat ground.
"""
import os

os.environ["PYTORCH_NVFUSER_DISABLE_FALLBACK"] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

HEADLESS = False
RECORD_VIDEO = True
EXPERIMENT_NAME = "quadruped_gait"
VIDEO_FPS = 30
NUM_ENVS = 16
CHECKPOINT = "model_499.pt"
CAMERA_HEIGHT = 3.0
CAMERA_DISTANCE = 5.0
EVALUATION_DURATION = 20.0

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

from rl_cfg import QuadrupedEnvCfg
from sim_cfg import FlatGroundSceneCfg
from evaluate_base import BaseEvaluator, set_camera_view


class FlatGroundEvaluator(BaseEvaluator):
    """Evaluator for flat ground environments."""

    def __init__(self):
        super().__init__(
            experiment_name=EXPERIMENT_NAME,
            checkpoint=CHECKPOINT,
            headless=HEADLESS,
            record_video=RECORD_VIDEO,
            video_fps=VIDEO_FPS,
            evaluation_duration=EVALUATION_DURATION,
        )
        self.video_dir = f"{self.log_dir}/videos_flat"
        self.frames_dir = f"{self.log_dir}/frames_flat"

    def create_env_cfg(self) -> QuadrupedEnvCfg:
        env_cfg = QuadrupedEnvCfg()
        env_cfg.scene = FlatGroundSceneCfg(num_envs=NUM_ENVS, env_spacing=2.5)
        return env_cfg

    def setup_environment(self, isaac_env):
        pass

    def setup_camera(self, isaac_env):
        set_camera_view(
            isaac_env,
            eye=(CAMERA_DISTANCE, CAMERA_DISTANCE, CAMERA_HEIGHT),
            target=(0.0, 0.0, 0.0)
        )

    def get_video_filename(self) -> str:
        return "evaluation_flat.mp4"

    def print_header(self, checkpoint_path: str, env_cfg: QuadrupedEnvCfg):
        print("\n" + "=" * 80)
        print("QUADRUPED POLICY EVALUATION (FLAT GROUND)")
        print("=" * 80)
        print(f"Log directory: {self.log_dir}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Num environments: {NUM_ENVS}")
        print(f"Evaluation duration: {self.evaluation_duration}s")
        print(f"Headless: {self.headless}")
        print(f"Record video: {self.record_video}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    evaluator = FlatGroundEvaluator()
    evaluator.run(simulation_app)
