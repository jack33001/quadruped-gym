"""
Plotting utilities for quadruped evaluation metrics.
"""
import os
from dataclasses import dataclass, field

import torch
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PerformanceMetrics:
    """Container for recorded performance metrics for a single gait."""
    gait_name: str
    env_idx: int
    
    time: list = field(default_factory=list)
    x_velocity: list = field(default_factory=list)
    commanded_velocity: list = field(default_factory=list)
    roll_rate: list = field(default_factory=list)
    pitch_rate: list = field(default_factory=list)
    yaw_rate: list = field(default_factory=list)
    heading: list = field(default_factory=list)
    energy: list = field(default_factory=list)
    power: list = field(default_factory=list)
    joint_torques: list = field(default_factory=list)
    foot_contacts_actual: list = field(default_factory=list)
    foot_contacts_desired: list = field(default_factory=list)
    ride_height: list = field(default_factory=list)
    
    _cumulative_energy: float = 0.0


class PerformanceRecorder:
    """Records performance metrics during evaluation."""
    
    def __init__(self, gait_names: list, env_indices: list, device: torch.device):
        """Initialize recorder for specified gaits and environments.
        
        Args:
            gait_names: List of gait names to track.
            env_indices: List of environment indices (one per gait).
            device: Torch device.
        """
        self.device = device
        self.gait_names = gait_names
        self.env_indices = env_indices
        self.metrics = {}
        
        for gait_name, env_idx in zip(gait_names, env_indices):
            self.metrics[gait_name] = PerformanceMetrics(
                gait_name=gait_name,
                env_idx=env_idx
            )
    
    def record(self, env, sim_time: float):
        """Record metrics at current timestep.
        
        Args:
            env: The wrapped environment.
            sim_time: Current simulation time in seconds.
        """
        isaac_env = env.unwrapped
        robot = isaac_env.scene["robot"]
        gait_scheduler = env.gait_scheduler
        
        root_lin_vel = robot.data.root_lin_vel_w
        root_ang_vel = robot.data.root_ang_vel_w
        root_quat = robot.data.root_quat_w
        root_pos = robot.data.root_pos_w
        applied_torques = robot.data.applied_torque
        joint_vel = robot.data.joint_vel
        
        desired_contacts, _ = gait_scheduler.get_gait_obs()
        
        foot_contact_sensor = isaac_env.scene.sensors["foot_contact"]
        contact_forces = foot_contact_sensor.data.net_forces_w
        contact_threshold = 1.0
        actual_contacts = (torch.norm(contact_forces, dim=-1) > contact_threshold).float()
        
        cmd_manager = isaac_env.command_manager
        velocity_cmd = None
        if hasattr(cmd_manager, 'get_command'):
            velocity_cmd = cmd_manager.get_command("base_velocity")
        
        for gait_name, env_idx in zip(self.gait_names, self.env_indices):
            metric = self.metrics[gait_name]
            idx = env_idx
            
            metric.time.append(sim_time)
            
            metric.x_velocity.append(root_lin_vel[idx, 0].item())
            
            if velocity_cmd is not None:
                metric.commanded_velocity.append(velocity_cmd[idx, 0].item())
            else:
                metric.commanded_velocity.append(0.0)
            
            metric.roll_rate.append(root_ang_vel[idx, 0].item())
            metric.pitch_rate.append(root_ang_vel[idx, 1].item())
            metric.yaw_rate.append(root_ang_vel[idx, 2].item())
            
            w, x, y, z = root_quat[idx, 0], root_quat[idx, 1], root_quat[idx, 2], root_quat[idx, 3]
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            metric.heading.append(yaw.item())
            
            torques = applied_torques[idx].cpu().numpy()
            vel = joint_vel[idx].cpu().numpy()
            power = np.sum(np.abs(torques * vel))
            metric.power.append(power)
            
            dt = metric.time[-1] - metric.time[-2] if len(metric.time) > 1 else 0.02
            metric._cumulative_energy += power * dt
            metric.energy.append(metric._cumulative_energy)
            
            metric.joint_torques.append(torques.copy())
            
            metric.foot_contacts_actual.append(actual_contacts[idx].cpu().numpy().copy())
            metric.foot_contacts_desired.append(desired_contacts[idx].cpu().numpy().copy())
            
            terrain_height = 0.0
            if hasattr(isaac_env.scene, 'terrain') and isaac_env.scene.terrain is not None:
                terrain = isaac_env.scene.terrain
                if hasattr(terrain, 'env_origins'):
                    terrain_height = terrain.env_origins[idx, 2].item()
            height = root_pos[idx, 2].item() - terrain_height
            metric.ride_height.append(height)
    
    def reset_gait(self, gait_name: str):
        """Reset metrics for a specific gait after episode reset."""
        if gait_name in self.metrics:
            m = self.metrics[gait_name]
            m.time.clear()
            m.x_velocity.clear()
            m.commanded_velocity.clear()
            m.roll_rate.clear()
            m.pitch_rate.clear()
            m.yaw_rate.clear()
            m.heading.clear()
            m.energy.clear()
            m.power.clear()
            m.joint_torques.clear()
            m.foot_contacts_actual.clear()
            m.foot_contacts_desired.clear()
            m.ride_height.clear()
            m._cumulative_energy = 0.0


class PerformancePlotter:
    """Generates plots from recorded performance metrics."""
    
    FOOT_NAMES = ["FR", "FL", "RR", "RL"]
    JOINT_NAMES = ["FR_Hip", "FR_Knee", "FL_Hip", "FL_Knee", 
                   "RR_Hip", "RR_Knee", "RL_Hip", "RL_Knee"]
    
    def __init__(self, output_dir: str):
        """Initialize plotter.
        
        Args:
            output_dir: Directory to save plots.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _compute_step_frequency(self, contacts: np.ndarray, time: np.ndarray) -> float:
        """Compute step frequency from contact states."""
        if len(contacts) < 2:
            return 0.0
        
        contact_diff = np.diff(contacts)
        liftoffs = np.sum(contact_diff < 0)
        
        duration = time[-1] - time[0]
        if duration <= 0:
            return 0.0
        
        return liftoffs / duration
    
    def _compute_duty_factor(self, contacts: np.ndarray) -> float:
        """Compute duty factor (fraction of time in stance) from contact states."""
        if len(contacts) < 2:
            return 0.0
        return np.mean(contacts)
    
    def plot_gait_metrics(self, metrics: PerformanceMetrics):
        """Generate combined plot for a single gait.
        
        Args:
            metrics: PerformanceMetrics instance with recorded data.
        """
        if len(metrics.time) < 2:
            print(f"Not enough data for gait '{metrics.gait_name}', skipping plots")
            return
        
        time = np.array(metrics.time)
        
        avg_cmd_vel = np.mean(metrics.commanded_velocity) if metrics.commanded_velocity else 0.0
        
        fig = plt.figure(figsize=(16, 20))
        gs = fig.add_gridspec(6, 1, hspace=0.3)
        
        fig.suptitle(f"Gait: {metrics.gait_name} | Commanded Velocity: {avg_cmd_vel:.2f} m/s", 
                     fontsize=14, fontweight='bold')
        
        avg_actual_vel = np.mean(metrics.x_velocity)
        if abs(avg_cmd_vel) > 0.01:
            vel_error_pct = abs(avg_actual_vel - avg_cmd_vel) / abs(avg_cmd_vel) * 100
        else:
            vel_error_pct = 0.0
        
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(time, metrics.x_velocity, 'b-', linewidth=1.5, label='Actual')
        ax1.plot(time, metrics.commanded_velocity, 'r--', linewidth=1.5, label='Commanded')
        ax1.set_ylabel('X Velocity (m/s)')
        ax1.set_title(f'Forward Velocity | Avg: {avg_actual_vel:.2f} m/s, Error: {vel_error_pct:.1f}%')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        max_roll = np.max(np.abs(metrics.roll_rate))
        max_pitch = np.max(np.abs(metrics.pitch_rate))
        max_yaw = np.max(np.abs(metrics.yaw_rate))
        
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(time, metrics.roll_rate, 'r-', label='Roll', linewidth=1.5)
        ax2.plot(time, metrics.pitch_rate, 'g-', label='Pitch', linewidth=1.5)
        ax2.plot(time, metrics.yaw_rate, 'b-', label='Yaw', linewidth=1.5)
        ax2.set_ylabel('Angular Rate (rad/s)')
        ax2.set_title(f'Angular Rates | Max Roll: {max_roll:.2f}, Pitch: {max_pitch:.2f}, Yaw: {max_yaw:.2f} rad/s')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        heading_deg = np.degrees(metrics.heading)
        heading_drift = heading_deg[-1] - heading_deg[0] if len(heading_deg) > 0 else 0.0
        
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(time, heading_deg, 'b-', linewidth=1.5)
        ax3.set_ylabel('Heading (deg)')
        ax3.set_title(f'Heading | Total Drift: {heading_drift:.1f} deg')
        ax3.grid(True, alpha=0.3)
        
        total_energy = metrics.energy[-1] if metrics.energy else 0
        avg_power = np.mean(metrics.power) if metrics.power else 0
        max_power = np.max(metrics.power) if metrics.power else 0
        
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax4.plot(time, metrics.power, 'b-', linewidth=1.5)
        ax4.set_ylabel('Power (W)')
        ax4.set_title(f'Power | Avg: {avg_power:.1f} W, Max: {max_power:.1f} W, Total Energy: {total_energy:.1f} J')
        ax4.grid(True, alpha=0.3)
        
        avg_torque = 0.0
        if metrics.joint_torques:
            torques = np.array(metrics.joint_torques)
            avg_torque = np.mean(np.abs(torques))
        
        ax5 = fig.add_subplot(gs[4], sharex=ax1)
        if metrics.joint_torques:
            torques = np.array(metrics.joint_torques)
            num_joints = torques.shape[1]
            colors = plt.cm.tab10(np.linspace(0, 1, num_joints))
            for j in range(num_joints):
                label = self.JOINT_NAMES[j] if j < len(self.JOINT_NAMES) else f'Joint {j}'
                ax5.plot(time, torques[:, j], color=colors[j], linewidth=1.0, label=label)
            ax5.legend(loc='upper right', ncol=2, fontsize=8)
        ax5.set_ylabel('Torque (Nm)')
        ax5.set_title(f'Joint Torques | Avg Abs Torque: {avg_torque:.2f} Nm')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        actual_freqs = []
        desired_freqs = []
        actual_duty_factors = []
        desired_duty_factors = []
        if metrics.foot_contacts_actual and metrics.foot_contacts_desired:
            actual = np.array(metrics.foot_contacts_actual)
            desired = np.array(metrics.foot_contacts_desired)
            num_feet = actual.shape[1]
            
            for foot_idx in range(num_feet):
                actual_freq = self._compute_step_frequency(actual[:, foot_idx], time)
                desired_freq = self._compute_step_frequency(desired[:, foot_idx], time)
                actual_freqs.append(actual_freq)
                desired_freqs.append(desired_freq)
                
                actual_duty = self._compute_duty_factor(actual[:, foot_idx])
                desired_duty = self._compute_duty_factor(desired[:, foot_idx])
                actual_duty_factors.append(actual_duty)
                desired_duty_factors.append(desired_duty)
        
        freq_str_parts = []
        for i, name in enumerate(self.FOOT_NAMES[:len(actual_freqs)]):
            freq_str_parts.append(f"{name}: {actual_freqs[i]:.1f}/{desired_freqs[i]:.1f}")
        freq_str = ", ".join(freq_str_parts)
        
        duty_str_parts = []
        for i, name in enumerate(self.FOOT_NAMES[:len(actual_duty_factors)]):
            duty_str_parts.append(f"{name}: {actual_duty_factors[i]*100:.0f}/{desired_duty_factors[i]*100:.0f}%")
        duty_str = ", ".join(duty_str_parts)
        
        ax6 = fig.add_subplot(gs[5], sharex=ax1)
        if metrics.foot_contacts_actual and metrics.foot_contacts_desired:
            actual = np.array(metrics.foot_contacts_actual)
            desired = np.array(metrics.foot_contacts_desired)
            num_feet = actual.shape[1]
            
            for foot_idx in range(num_feet):
                foot_name = self.FOOT_NAMES[foot_idx] if foot_idx < len(self.FOOT_NAMES) else f'Foot {foot_idx}'
                offset = foot_idx * 1.5
                
                ax6.fill_between(time, offset, offset + desired[:, foot_idx], 
                               alpha=0.3, color='blue', step='post')
                ax6.step(time, offset + actual[:, foot_idx], where='post', 
                        color='red', linewidth=1.5)
                
                ax6.text(-0.02, offset + 0.5, foot_name, transform=ax6.get_yaxis_transform(),
                        ha='right', va='center', fontsize=9)
            
            ax6.set_ylim(-0.2, num_feet * 1.5 + 0.2)
            ax6.set_yticks([])
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', alpha=0.3, label='Desired Stance'),
                Patch(facecolor='red', alpha=0.8, label='Actual Contact')
            ]
            ax6.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        ax6.set_xlabel('Time (s)')
        ax6.set_title(f'Foot Contact States | Freq (Hz): {freq_str}\nDuty Factor: {duty_str}')
        ax6.grid(True, alpha=0.3, axis='x')
        
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_xticklabels(), visible=False)
        plt.setp(ax4.get_xticklabels(), visible=False)
        plt.setp(ax5.get_xticklabels(), visible=False)
        
        output_path = os.path.join(self.output_dir, f"gait_{metrics.gait_name}.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved plot for gait '{metrics.gait_name}' to {output_path}")
    
    def plot_all(self, recorder: PerformanceRecorder):
        """Generate plots for all recorded gaits.
        
        Args:
            recorder: PerformanceRecorder with recorded metrics.
        """
        for gait_name, metrics in recorder.metrics.items():
            self.plot_gait_metrics(metrics)
