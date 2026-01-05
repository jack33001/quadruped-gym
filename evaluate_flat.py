"""
Evaluation script for trained quadruped policies on flat ground.

Runs headless with 1000 environments per gait, generating comprehensive
performance plots and a summary table.
"""
import os
import shutil

os.environ["PYTORCH_NVFUSER_DISABLE_FALLBACK"] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

import matplotlib
matplotlib.use('Agg')

from isaaclab.app import AppLauncher

from train_cfg import TrainCfg

TRAIN_CFG = TrainCfg()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from isaaclab.envs import ManagerBasedRLEnv

from rl_cfg import QuadrupedEnvCfg
from sim_cfg import FlatGroundSceneCfg
from quadruped_env import IsaacLabVecEnvWrapper
from evaluate_base import find_checkpoint, load_policy
from gait_cfg import GaitType, GAIT_PARAMS, GaitSchedulerCfg


@dataclass
class GaitStatistics:
    """Statistics for a single gait evaluation."""
    gait_name: str
    
    energy_samples: list = field(default_factory=list)
    velocity_error_samples: list = field(default_factory=list)
    gait_timing_error_samples: list = field(default_factory=list)
    heading_drift_samples: list = field(default_factory=list)
    joint_torque_samples: list = field(default_factory=list)
    pitch_rate_samples: list = field(default_factory=list)
    roll_rate_samples: list = field(default_factory=list)
    
    time_series_velocity: list = field(default_factory=list)
    time_series_commanded: list = field(default_factory=list)
    time_series_power: list = field(default_factory=list)
    time_series_roll_rate: list = field(default_factory=list)
    time_series_pitch_rate: list = field(default_factory=list)
    time_series_heading: list = field(default_factory=list)
    time_series_height: list = field(default_factory=list)
    time_series_time: list = field(default_factory=list)
    
    transient_samples: int = 0
    
    def set_transient_samples(self, num_samples: int):
        """Set number of initial samples to skip for steady-state calculations."""
        self.transient_samples = num_samples
    
    def _get_steady_state(self, samples: list) -> list:
        """Return samples after transient period."""
        if len(samples) <= self.transient_samples:
            return samples
        return samples[self.transient_samples:]
    
    def avg_energy(self) -> float:
        samples = self._get_steady_state(self.energy_samples)
        return np.mean(samples) if samples else 0.0
    
    def avg_velocity_error(self) -> float:
        samples = self._get_steady_state(self.velocity_error_samples)
        return np.mean(samples) if samples else 0.0
    
    def avg_gait_timing_error(self) -> float:
        samples = self._get_steady_state(self.gait_timing_error_samples)
        return np.mean(samples) if samples else 0.0
    
    def avg_heading_drift(self) -> float:
        samples = self._get_steady_state(self.heading_drift_samples)
        return np.mean(samples) if samples else 0.0
    
    def std_joint_torque(self) -> float:
        samples = self._get_steady_state(self.joint_torque_samples)
        if not samples:
            return 0.0
        all_torques = np.concatenate(samples)
        return np.std(all_torques)
    
    def avg_pitch_rate(self) -> float:
        samples = self._get_steady_state(self.pitch_rate_samples)
        return np.mean(np.abs(samples)) if samples else 0.0
    
    def avg_roll_rate(self) -> float:
        samples = self._get_steady_state(self.roll_rate_samples)
        return np.mean(np.abs(samples)) if samples else 0.0


class FlatGroundEvaluator:
    """Evaluator for flat ground with comprehensive statistics."""
    
    def __init__(self, train_cfg: TrainCfg):
        self.train_cfg = train_cfg
        self.log_dir = f"logs/{train_cfg.experiment_name}"
        self.output_dir = f"{self.log_dir}/eval_flat"
        self.plots_dir = f"{self.output_dir}/plots"
        
        self.num_envs_per_gait = 1000
        self.evaluation_duration = 10.0
        self.transient_duration = 1.0
        
        gait_cfg = GaitSchedulerCfg()
        self.enabled_gaits = list(gait_cfg.enabled_gaits)
        
        self.gait_stats = {}
        for gait_type in self.enabled_gaits:
            gait_name = GAIT_PARAMS[gait_type].name
            self.gait_stats[gait_name] = GaitStatistics(gait_name=gait_name)
        
        self.termination_counts = {}

    def setup_dirs(self):
        """Create output directories, clearing previous eval results."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def create_env(self) -> tuple:
        """Create environment."""
        env_cfg = QuadrupedEnvCfg()
        env_cfg.scene = FlatGroundSceneCfg(
            num_envs=self.num_envs_per_gait, 
            env_spacing=2.5
        )
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.5, 2.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        
        isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
        env = IsaacLabVecEnvWrapper(isaac_env)
        
        return env, isaac_env, env_cfg
    
    def set_gait_for_all_envs(self, env, gait_type: GaitType):
        """Set the same gait type for all environments."""
        env.gait_scheduler.gait_types.fill_(gait_type)
        env.gait_scheduler.cfg.allow_gait_switching = False
    
    def evaluate_gait(self, env, isaac_env, policy, gait_type: GaitType, sim_dt: float):
        """Evaluate a single gait across all environments."""
        gait_name = GAIT_PARAMS[gait_type].name
        stats = self.gait_stats[gait_name]
        
        transient_steps = int(self.transient_duration / sim_dt)
        samples_per_step = env.num_envs
        stats.set_transient_samples(transient_steps * samples_per_step)
        
        print(f"  Evaluating {gait_name}...")
        print(f"    Skipping first {self.transient_duration}s ({transient_steps} steps) for steady-state averages")
        
        self.set_gait_for_all_envs(env, gait_type)
        obs_dict = env.reset()
        self.set_gait_for_all_envs(env, gait_type)
        
        total_steps = int(self.evaluation_duration / sim_dt)
        
        cumulative_energy = torch.zeros(env.num_envs, device=env.device)
        initial_heading = None
        step_count = 0
        termination_count = 0
        
        record_interval = max(1, total_steps // 100)
        
        print(f"    Running {total_steps} steps across {env.num_envs} environments...")
        
        cmd_manager = isaac_env.command_manager
        velocity_cmd = cmd_manager.get_command("base_velocity")
        if velocity_cmd is not None:
            print(f"    Initial velocity commands - min: {velocity_cmd[:, 0].min().item():.2f}, max: {velocity_cmd[:, 0].max().item():.2f}, mean: {velocity_cmd[:, 0].mean().item():.2f}")
        
        for step in range(total_steps):
            with torch.no_grad():
                actions = policy.act_inference(obs_dict)
            
            obs_dict, _, dones, _ = env.step(actions)
            step_count += 1
            
            robot = isaac_env.scene["robot"]
            cmd_manager = isaac_env.command_manager
            
            root_lin_vel = robot.data.root_lin_vel_w
            root_ang_vel = robot.data.root_ang_vel_w
            root_quat = robot.data.root_quat_w
            root_pos = robot.data.root_pos_w
            applied_torques = robot.data.applied_torque
            joint_vel = robot.data.joint_vel
            
            velocity_cmd = cmd_manager.get_command("base_velocity")
            
            if step == 0:
                print(f"    Step 0 - Actions mean: {actions.mean().item():.4f}, std: {actions.std().item():.4f}")
                print(f"    Step 0 - Actual velocity mean: {root_lin_vel[:, 0].mean().item():.4f}")
                print(f"    Step 0 - Commanded velocity mean: {velocity_cmd[:, 0].mean().item():.4f}")
            
            w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
            current_heading = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            
            if initial_heading is None:
                initial_heading = current_heading.clone()
            
            power = torch.sum(torch.abs(applied_torques * joint_vel), dim=-1)
            cumulative_energy += power * sim_dt
            
            actual_vel = root_lin_vel[:, 0]
            commanded_vel = velocity_cmd[:, 0]
            vel_error = torch.abs(actual_vel - commanded_vel) / commanded_vel.clamp(min=0.1) * 100
            
            desired_contacts = env.gait_scheduler.get_contact_states()
            foot_contact_sensor = isaac_env.scene.sensors["foot_contact"]
            contact_forces = foot_contact_sensor.data.net_forces_w
            actual_contacts = (torch.norm(contact_forces, dim=-1) > 1.0).float()
            gait_error = torch.mean(torch.abs(desired_contacts - actual_contacts), dim=-1)
            
            heading_drift = torch.abs(current_heading - initial_heading)
            heading_drift = torch.where(heading_drift > 3.14159, 2 * 3.14159 - heading_drift, heading_drift)
            heading_drift_deg = heading_drift * (180.0 / 3.14159265359)
            
            roll_rate = root_ang_vel[:, 0]
            pitch_rate = root_ang_vel[:, 1]
            
            stats.energy_samples.extend(cumulative_energy.cpu().numpy().tolist())
            stats.velocity_error_samples.extend(vel_error.cpu().numpy().tolist())
            stats.gait_timing_error_samples.extend(gait_error.cpu().numpy().tolist())
            stats.heading_drift_samples.extend(heading_drift_deg.cpu().numpy().tolist())
            stats.joint_torque_samples.append(applied_torques.cpu().numpy().flatten())
            stats.pitch_rate_samples.extend(pitch_rate.cpu().numpy().tolist())
            stats.roll_rate_samples.extend(roll_rate.cpu().numpy().tolist())
            
            if step % record_interval == 0:
                current_heading_deg = current_heading * (180.0 / 3.14159265359)
                stats.time_series_time.append(step * sim_dt)
                stats.time_series_velocity.append(actual_vel.mean().item())
                stats.time_series_commanded.append(commanded_vel.mean().item())
                stats.time_series_power.append(power.mean().item())
                stats.time_series_roll_rate.append(roll_rate.mean().item())
                stats.time_series_pitch_rate.append(pitch_rate.mean().item())
                stats.time_series_heading.append(current_heading_deg.mean().item())
                stats.time_series_height.append(root_pos[:, 2].mean().item())
            
            if dones.any():
                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_indices.dim() == 0:
                    done_indices = done_indices.unsqueeze(0)
                termination_count += len(done_indices)
                cumulative_energy[done_indices] = 0.0
                initial_heading[done_indices] = current_heading[done_indices]
        
        print(f"    Final velocity mean: {root_lin_vel[:, 0].mean().item():.4f}")
        self.termination_counts[gait_name] = termination_count
        print(f"    Completed {step_count} steps, {termination_count} early terminations")
    
    def generate_summary_table(self) -> str:
        """Generate summary table as formatted string and save to file."""
        gait_names = [GAIT_PARAMS[gt].name for gt in self.enabled_gaits]
        
        headers = ["Metric"] + gait_names
        rows = [
            ["Avg Energy (J)"] + [f"{self.gait_stats[g].avg_energy():.2f}" for g in gait_names],
            ["Avg Velocity Error (%)"] + [f"{self.gait_stats[g].avg_velocity_error():.1f}" for g in gait_names],
            ["Avg Gait Timing Error (0-1)"] + [f"{self.gait_stats[g].avg_gait_timing_error():.3f}" for g in gait_names],
            ["Avg Heading Drift (deg)"] + [f"{self.gait_stats[g].avg_heading_drift():.2f}" for g in gait_names],
            ["Joint Torque Std (Nm)"] + [f"{self.gait_stats[g].std_joint_torque():.3f}" for g in gait_names],
            ["Avg Pitch Rate (rad/s)"] + [f"{self.gait_stats[g].avg_pitch_rate():.3f}" for g in gait_names],
            ["Avg Roll Rate (rad/s)"] + [f"{self.gait_stats[g].avg_roll_rate():.3f}" for g in gait_names],
            ["Early Terminations"] + [f"{self.termination_counts.get(g, 0)}" for g in gait_names],
        ]
        
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
        
        def format_row(row):
            return " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        
        lines = []
        lines.append(format_row(headers))
        lines.append("-" * (sum(col_widths) + 3 * (len(headers) - 1)))
        for row in rows:
            lines.append(format_row(row))
        
        table_str = "\n".join(lines)
        
        table_path = os.path.join(self.output_dir, "summary_table.txt")
        with open(table_path, 'w') as f:
            f.write(table_str)
        
        return table_str
    
    def generate_per_gait_plots(self):
        """Generate detailed plot for each gait."""
        print(f"    Generating plots in: {self.plots_dir}")
        
        for gait_type in self.enabled_gaits:
            gait_name = GAIT_PARAMS[gait_type].name
            stats = self.gait_stats[gait_name]
            
            if not stats.time_series_time:
                print(f"    No time series data for {gait_name}, skipping")
                continue
            
            print(f"    Generating plot for {gait_name} ({len(stats.time_series_time)} data points)")
            
            time = np.array(stats.time_series_time)
            
            fig = plt.figure(figsize=(16, 20))
            fig.suptitle(f"Gait: {gait_name.upper()}", fontsize=16, fontweight='bold')
            
            ax1 = fig.add_subplot(6, 1, 1)
            ax1.plot(time, stats.time_series_velocity, 'b-', linewidth=1.5, label='Actual')
            ax1.plot(time, stats.time_series_commanded, 'r--', linewidth=1.5, label='Commanded')
            ax1.set_ylabel('Velocity (m/s)')
            ax1.set_title(f'Forward Velocity | Avg Error: {stats.avg_velocity_error():.1f}%')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            ax2 = fig.add_subplot(6, 1, 2)
            ax2.plot(time, stats.time_series_roll_rate, 'r-', linewidth=1.5, label='Roll')
            ax2.plot(time, stats.time_series_pitch_rate, 'g-', linewidth=1.5, label='Pitch')
            ax2.set_ylabel('Angular Rate (rad/s)')
            ax2.set_title(f'Angular Rates | Avg Roll: {stats.avg_roll_rate():.3f}, Avg Pitch: {stats.avg_pitch_rate():.3f} rad/s')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            ax3 = fig.add_subplot(6, 1, 3)
            ax3.plot(time, stats.time_series_heading, 'b-', linewidth=1.5)
            ax3.set_ylabel('Heading (deg)')
            ax3.set_title(f'Heading | Avg Drift: {stats.avg_heading_drift():.2f} deg')
            ax3.grid(True, alpha=0.3)
            
            ax4 = fig.add_subplot(6, 1, 4)
            ax4.plot(time, stats.time_series_height, 'b-', linewidth=1.5)
            ax4.axhline(y=0.22, color='r', linestyle='--', linewidth=1.0, label='Target')
            ax4.set_ylabel('Height (m)')
            ax4.set_title('Ride Height')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            
            ax5 = fig.add_subplot(6, 1, 5)
            ax5.plot(time, stats.time_series_power, 'b-', linewidth=1.5)
            ax5.set_ylabel('Power (W)')
            ax5.set_title(f'Power | Total Energy: {stats.avg_energy():.2f} J')
            ax5.grid(True, alpha=0.3)
            
            ax6 = fig.add_subplot(6, 1, 6)
            metrics = ['Vel Error', 'Gait Error', 'Heading Drift', 'Torque Std']
            values = [
                stats.avg_velocity_error() / 100,
                stats.avg_gait_timing_error(),
                stats.avg_heading_drift() / 90,
                stats.std_joint_torque() / 10,
            ]
            bar_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
            ax6.bar(metrics, values, color=bar_colors)
            ax6.set_ylabel('Normalized Value')
            ax6.set_title('Performance Summary (Normalized)')
            ax6.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            output_path = os.path.join(self.plots_dir, f"gait_{gait_name}.png")
            try:
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"    Saved plot: {output_path}")
            except Exception as e:
                print(f"    Error saving plot: {e}")
            plt.close(fig)
    
    def generate_comparison_plots(self):
        """Generate comparison plots across all gaits."""
        gait_names = [GAIT_PARAMS[gt].name for gt in self.enabled_gaits]
        colors = plt.cm.tab10(np.linspace(0, 1, len(gait_names)))
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle('Gait Comparison', fontsize=14, fontweight='bold')
        
        ax = axes[0, 0]
        for i, gait_name in enumerate(gait_names):
            stats = self.gait_stats[gait_name]
            if stats.time_series_time:
                ax.plot(stats.time_series_time, stats.time_series_velocity, 
                       color=colors[i], linewidth=1.5, label=gait_name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Forward Velocity')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        for i, gait_name in enumerate(gait_names):
            stats = self.gait_stats[gait_name]
            if stats.time_series_time:
                ax.plot(stats.time_series_time, stats.time_series_power, 
                       color=colors[i], linewidth=1.5, label=gait_name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power (W)')
        ax.set_title('Power Consumption')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        for i, gait_name in enumerate(gait_names):
            stats = self.gait_stats[gait_name]
            if stats.time_series_time:
                ax.plot(stats.time_series_time, stats.time_series_roll_rate, 
                       color=colors[i], linewidth=1.5, label=gait_name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Roll Rate (rad/s)')
        ax.set_title('Roll Rate')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        for i, gait_name in enumerate(gait_names):
            stats = self.gait_stats[gait_name]
            if stats.time_series_time:
                ax.plot(stats.time_series_time, stats.time_series_pitch_rate, 
                       color=colors[i], linewidth=1.5, label=gait_name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pitch Rate (rad/s)')
        ax.set_title('Pitch Rate')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        ax = axes[2, 0]
        for i, gait_name in enumerate(gait_names):
            stats = self.gait_stats[gait_name]
            if stats.time_series_time:
                ax.plot(stats.time_series_time, stats.time_series_heading, 
                       color=colors[i], linewidth=1.5, label=gait_name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heading (deg)')
        ax.set_title('Heading')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        ax = axes[2, 1]
        for i, gait_name in enumerate(gait_names):
            stats = self.gait_stats[gait_name]
            if stats.time_series_time:
                ax.plot(stats.time_series_time, stats.time_series_height, 
                       color=colors[i], linewidth=1.5, label=gait_name)
        ax.axhline(y=0.22, color='k', linestyle='--', linewidth=1.0, label='Target')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Height (m)')
        ax.set_title('Ride Height')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.plots_dir, "gait_comparison_timeseries.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved comparison plot: {output_path}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ['Energy\n(J)', 'Vel Error\n(%)', 'Gait Error\n(x100)', 
                   'Heading Drift\n(deg)', 'Torque Std\n(Nm)', 
                   'Pitch Rate\n(rad/s)', 'Roll Rate\n(rad/s)']
        
        x = np.arange(len(metrics))
        width = 0.8 / len(gait_names)
        
        for i, gait_name in enumerate(gait_names):
            stats = self.gait_stats[gait_name]
            values = [
                stats.avg_energy(),
                stats.avg_velocity_error(),
                stats.avg_gait_timing_error() * 100,
                stats.avg_heading_drift(),
                stats.std_joint_torque(),
                stats.avg_pitch_rate(),
                stats.avg_roll_rate(),
            ]
            offset = (i - len(gait_names) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=gait_name, color=colors[i])
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        ax.set_title('Performance Metrics by Gait')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.plots_dir, "gait_comparison_bar.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved bar chart: {output_path}")
    
    def run(self):
        """Run the full evaluation."""
        self.setup_dirs()
        
        if not os.path.exists(self.log_dir):
            raise ValueError(f"Log directory {self.log_dir} not found")
        
        checkpoint_path = find_checkpoint(self.log_dir, f"model_{self.train_cfg.max_iterations - 1}.pt")
        
        print("\n" + "=" * 80)
        print("FLAT GROUND EVALUATION")
        print("=" * 80)
        print(f"Log directory: {self.log_dir}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Environments per gait: {self.num_envs_per_gait}")
        print(f"Evaluation duration: {self.evaluation_duration}s")
        print(f"Gaits: {[GAIT_PARAMS[gt].name for gt in self.enabled_gaits]}")
        print("=" * 80 + "\n")
        
        env, isaac_env, env_cfg = self.create_env()
        sim_dt = env_cfg.sim.dt
        
        obs_dict = env.reset()
        policy = load_policy(self.log_dir, checkpoint_path, obs_dict, env.num_actions, env.device)
        print("Policy loaded successfully\n")
        
        print("Evaluating gaits...")
        for gait_type in self.enabled_gaits:
            self.evaluate_gait(env, isaac_env, policy, gait_type, sim_dt)
        
        print("\nGenerating summary table...")
        table_str = self.generate_summary_table()
        print("\n" + table_str + "\n")
        
        print("Generating per-gait plots...")
        self.generate_per_gait_plots()
        
        print("Generating comparison plots...")
        self.generate_comparison_plots()
        
        print(f"\nResults saved to: {self.output_dir}")
        print("Evaluation complete!")
        
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    evaluator = FlatGroundEvaluator(TRAIN_CFG)
    evaluator.run()
