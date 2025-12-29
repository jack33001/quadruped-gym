"""
Evaluation script for trained quadruped policies on flat ground.
"""
import os

os.environ["PYTORCH_NVFUSER_DISABLE_FALLBACK"] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

from isaaclab.app import AppLauncher

from train_cfg import TrainCfg

TRAIN_CFG = TrainCfg()
EVAL_CFG = TRAIN_CFG.eval

app_launcher = AppLauncher(headless=EVAL_CFG.headless)
simulation_app = app_launcher.app

from rl_cfg import QuadrupedEnvCfg
from sim_cfg import FlatGroundSceneCfg
from evaluate_base import BaseEvaluator, set_camera_view
from plot_utils import PerformanceRecorder
from gait_cfg import GaitType, GAIT_PARAMS


class FlatGroundEvaluator(BaseEvaluator):
    """Evaluator for flat ground environments."""

    def __init__(self, train_cfg: TrainCfg):
        eval_cfg = train_cfg.eval
        super().__init__(
            experiment_name=train_cfg.experiment_name,
            checkpoint=f"model_{train_cfg.max_iterations - 1}.pt",
            headless=eval_cfg.headless,
            record_video=eval_cfg.record_video,
            video_fps=eval_cfg.video_fps,
            evaluation_duration=eval_cfg.evaluation_duration,
        )
        self.eval_cfg = eval_cfg
        self.video_dir = f"{self.log_dir}/videos"
        self.frames_dir = f"{self.log_dir}/frames"
        self.plots_dir = f"{self.log_dir}/plots_flat"

    def create_env_cfg(self) -> QuadrupedEnvCfg:
        env_cfg = QuadrupedEnvCfg()
        num_gaits = len(GaitType)
        num_envs = max(self.eval_cfg.flat_ground_num_envs, num_gaits)
        env_cfg.scene = FlatGroundSceneCfg(
            num_envs=num_envs, 
            env_spacing=2.5
        )
        return env_cfg

    def setup_environment(self, isaac_env):
        pass

    def setup_camera(self, isaac_env):
        set_camera_view(
            isaac_env,
            eye=(self.eval_cfg.camera_distance, self.eval_cfg.camera_distance, self.eval_cfg.camera_height),
            target=(0.0, 0.0, 0.0)
        )

    def setup_performance_recording(self, env) -> PerformanceRecorder:
        """Setup recording with one environment per gait type."""
        gait_scheduler = env.gait_scheduler
        
        all_gait_names = [GAIT_PARAMS[gt].name for gt in GaitType]
        num_gaits = len(all_gait_names)
        
        for i in range(min(num_gaits, env.num_envs)):
            gait_scheduler.gait_types[i] = i
        
        env_indices = list(range(min(num_gaits, env.num_envs)))
        gait_names = all_gait_names[:len(env_indices)]
        
        print(f"Recording performance for gaits: {gait_names}")
        print(f"  Environment indices: {env_indices}")
        
        return PerformanceRecorder(gait_names, env_indices, env.device)

    def get_video_filename(self) -> str:
        return "evaluation_flat.mp4"

    def print_header(self, checkpoint_path: str, env_cfg: QuadrupedEnvCfg):
        print("\n" + "=" * 80)
        print("QUADRUPED POLICY EVALUATION (FLAT GROUND)")
        print("=" * 80)
        print(f"Log directory: {self.log_dir}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Num environments: {env_cfg.scene.num_envs}")
        print(f"Evaluation duration: {self.evaluation_duration}s")
        print(f"Headless: {self.headless}")
        print(f"Record video: {self.record_video}")
        print(f"Performance plots will be saved to: {self.plots_dir}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    evaluator = FlatGroundEvaluator(TRAIN_CFG)
    evaluator.run(simulation_app)
