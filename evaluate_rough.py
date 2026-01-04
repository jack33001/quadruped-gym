"""
Evaluation script for trained quadruped policies on rough terrain.
"""
import os
import math

os.environ["PYTORCH_NVFUSER_DISABLE_FALLBACK"] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

from isaaclab.app import AppLauncher

from train_cfg import TrainCfg

TRAIN_CFG = TrainCfg()
EVAL_CFG = TRAIN_CFG.eval

app_launcher = AppLauncher(headless=EVAL_CFG.headless)
simulation_app = app_launcher.app

import torch

from rl_cfg import QuadrupedEnvCfg
from sim_cfg import QuadrupedSceneCfg
from evaluate_base import BaseEvaluator, set_camera_view
from plot_utils import PerformanceRecorder
from gait_cfg import GaitType, GAIT_PARAMS


def get_terrain_bounds(env):
    """Get the bounds and cell info of the subterrain grid."""
    isaac_env = env.unwrapped

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

    grid_width = cell_size[0] * num_cols
    grid_height = cell_size[1] * num_rows

    return {
        "min_x": -grid_width / 2,
        "max_x": grid_width / 2,
        "min_y": -grid_height / 2,
        "max_y": grid_height / 2,
        "cell_size": cell_size,
        "num_rows": num_rows,
        "num_cols": num_cols,
    }


def create_spotlight(isaac_env, intensity: float):
    """Create an overhead spotlight."""
    try:
        import omni.usd
        from pxr import UsdLux, Gf

        stage = omni.usd.get_context().get_stage()
        light_path = "/World/eval_spotlight"
        light_prim = stage.DefinePrim(light_path, "SphereLight")

        sphere_light = UsdLux.SphereLight(light_prim)
        sphere_light.CreateIntensityAttr(intensity)
        sphere_light.CreateRadiusAttr(2.0)
        sphere_light.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.9))
        sphere_light.CreateEnableColorTemperatureAttr(False)

        print(f"Created overhead spotlight at {light_path}")
        return light_path
    except Exception as e:
        print(f"Warning: Could not create spotlight: {e}")
        return None


def update_spotlight(light_path: str, x: float, y: float, height: float):
    """Update spotlight position."""
    try:
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        light_prim = stage.GetPrimAtPath(light_path)

        if light_prim and light_prim.IsValid():
            xformable = UsdGeom.Xformable(light_prim)
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(x, y, height))
    except Exception:
        pass


def distribute_robots_across_terrain(isaac_env, num_rows: int, num_cols: int):
    """Distribute robots evenly across terrain cells."""
    num_cells = num_rows * num_cols
    num_envs = isaac_env.num_envs

    terrain = isaac_env.scene.terrain
    if terrain is None:
        print("Warning: No terrain found")
        return False

    device = isaac_env.device

    terrain_levels = torch.zeros(num_envs, dtype=torch.long, device=device)
    terrain_types = torch.zeros(num_envs, dtype=torch.long, device=device)

    for env_idx in range(min(num_envs, num_cells)):
        row = env_idx // num_cols
        col = env_idx % num_cols
        terrain_levels[env_idx] = row
        terrain_types[env_idx] = col

    for env_idx in range(num_cells, num_envs):
        row = env_idx % num_rows
        col = env_idx % num_cols
        terrain_levels[env_idx] = row
        terrain_types[env_idx] = col

    terrain.terrain_levels = terrain_levels
    terrain.terrain_types = terrain_types

    if hasattr(terrain, 'terrain_origins'):
        for env_idx in range(num_envs):
            row = terrain_levels[env_idx].item()
            col = terrain_types[env_idx].item()
            if row < terrain.terrain_origins.shape[0] and col < terrain.terrain_origins.shape[1]:
                terrain.env_origins[env_idx] = terrain.terrain_origins[row, col]

    print(f"Distributed {num_envs} robots across {num_rows}x{num_cols} terrain grid")
    return True


class RoughTerrainEvaluator(BaseEvaluator):
    """Evaluator for rough terrain environments with camera panning."""

    def __init__(self, train_cfg: TrainCfg):
        eval_cfg = train_cfg.eval
        super().__init__(
            experiment_name=train_cfg.experiment_name,
            checkpoint=f"model_{train_cfg.max_iterations - 1}.pt",
            headless=eval_cfg.headless,
            record_video=eval_cfg.record_video,
            video_fps=eval_cfg.video_fps,
            evaluation_duration=eval_cfg.camera_pan_duration,
        )
        self.eval_cfg = eval_cfg
        self.pan_config = None
        self.spotlight_path = None
        self.num_rows = 6
        self.num_cols = 3
        self.plots_dir = f"{self.log_dir}/plots_rough"

    def create_env_cfg(self) -> QuadrupedEnvCfg:
        env_cfg = QuadrupedEnvCfg()
        env_cfg.scene = QuadrupedSceneCfg(num_envs=1, env_spacing=2.5)

        terrain_cfg = env_cfg.scene.terrain.terrain_generator
        self.num_rows = terrain_cfg.num_rows
        self.num_cols = terrain_cfg.num_cols
        num_cells = self.num_rows * self.num_cols

        env_cfg.scene.num_envs = num_cells
        terrain_cfg.curriculum = False
        env_cfg.scene.terrain.max_init_terrain_level = self.num_rows - 1

        return env_cfg

    def setup_environment(self, isaac_env):
        distribute_robots_across_terrain(isaac_env, self.num_rows, self.num_cols)
        self.spotlight_path = create_spotlight(isaac_env, self.eval_cfg.spotlight_intensity)

    def setup_camera(self, isaac_env):
        terrain = isaac_env.scene.terrain

        if hasattr(terrain, 'terrain_origins'):
            origins = terrain.terrain_origins
            first_cell = origins[0, 0].cpu().numpy()
            last_cell = origins[-1, -1].cpu().numpy()
            first_cell_x, first_cell_y = float(first_cell[0]), float(first_cell[1])
            last_cell_x, last_cell_y = float(last_cell[0]), float(last_cell[1])
            print(f"First cell origin: ({first_cell_x:.1f}, {first_cell_y:.1f})")
            print(f"Last cell origin: ({last_cell_x:.1f}, {last_cell_y:.1f})")
        else:
            bounds = get_terrain_bounds(self)
            cell_w, cell_h = bounds["cell_size"]
            first_cell_x = bounds["min_x"] + cell_w / 2
            first_cell_y = bounds["min_y"] + cell_h / 2
            last_cell_x = bounds["max_x"] - cell_w / 2
            last_cell_y = bounds["max_y"] - cell_h / 2

        dx = last_cell_x - first_cell_x
        dy = last_cell_y - first_cell_y
        dist = math.sqrt(dx * dx + dy * dy)
        dir_x = dx / dist if dist > 0.001 else 1.0
        dir_y = dy / dist if dist > 0.001 else 0.0

        cam_height = self.eval_cfg.camera_pan_height
        look_ahead = self.eval_cfg.camera_look_ahead
        target_height = self.eval_cfg.camera_target_height

        self.pan_config = {
            "start_pos": (first_cell_x, first_cell_y, cam_height),
            "end_pos": (last_cell_x - dir_x * look_ahead, last_cell_y - dir_y * look_ahead, cam_height),
            "look_dir": (dir_x, dir_y),
            "look_ahead": look_ahead,
            "target_height": target_height,
        }

        print(f"Camera pan: {self.pan_config['start_pos']} -> {self.pan_config['end_pos']}")

        set_camera_view(
            isaac_env,
            eye=self.pan_config["start_pos"],
            target=(
                self.pan_config["start_pos"][0] + dir_x * look_ahead,
                self.pan_config["start_pos"][1] + dir_y * look_ahead,
                target_height
            )
        )

        if self.spotlight_path:
            update_spotlight(self.spotlight_path, first_cell_x, first_cell_y, self.eval_cfg.spotlight_height)

    def update_camera(self, isaac_env, step_count: int, total_steps: int):
        if self.pan_config is None:
            return

        progress = step_count / total_steps
        start = self.pan_config["start_pos"]
        end = self.pan_config["end_pos"]
        look_dir = self.pan_config["look_dir"]

        eye_x = start[0] + (end[0] - start[0]) * progress
        eye_y = start[1] + (end[1] - start[1]) * progress
        eye_z = self.eval_cfg.camera_pan_height

        target_x = eye_x + look_dir[0] * self.eval_cfg.camera_look_ahead
        target_y = eye_y + look_dir[1] * self.eval_cfg.camera_look_ahead

        set_camera_view(isaac_env, eye=(eye_x, eye_y, eye_z), target=(target_x, target_y, self.eval_cfg.camera_target_height))

        if self.spotlight_path:
            update_spotlight(self.spotlight_path, eye_x, eye_y, self.eval_cfg.spotlight_height)

    def setup_performance_recording(self, env) -> PerformanceRecorder:
        """Setup recording with one environment per gait type on rough terrain."""
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
        return "evaluation.mp4"

    def print_header(self, checkpoint_path: str, env_cfg: QuadrupedEnvCfg):
        print("\n" + "=" * 80)
        print("QUADRUPED POLICY EVALUATION (ROUGH TERRAIN)")
        print("=" * 80)
        print(f"Log directory: {self.log_dir}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Terrain grid: {self.num_rows} rows x {self.num_cols} cols")
        print(f"Terrain mode: Random (non-curriculum)")
        print(f"Velocity range: 0.0 to 2.0 m/s")
        print(f"Camera pan duration: {self.evaluation_duration}s")
        print(f"Headless: {self.headless}")
        print(f"Record video: {self.record_video}")
        print(f"Performance plots will be saved to: {self.plots_dir}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    evaluator = RoughTerrainEvaluator(TRAIN_CFG)
    evaluator.run(simulation_app)
