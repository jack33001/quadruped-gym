"""
Test script to spawn the robot in its default configuration.
No resets, no policy - just observe the default spawn state.
"""
import os

os.environ["PYTORCH_NVFUSER_DISABLE_FALLBACK"] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from configs import QuadrupedEnvCfg


def main():
    """Spawn robot and observe default state."""
    
    # Create environment with single env
    env_cfg = QuadrupedEnvCfg()
    env_cfg.scene.num_envs = 1
    
    print("\n" + "=" * 80)
    print("ROBOT SPAWN TEST")
    print("=" * 80)
    print("Spawning robot in default configuration...")
    print("No resets, no policy actions - just observing default state")
    print("Press Ctrl+C to exit")
    print("=" * 80 + "\n")
    
    # Create environment (this spawns the robot)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Get robot asset
    robot = env.scene["robot"]
    
    # Print initial state
    print("Initial robot state:")
    print(f"  Root position: {robot.data.root_pos_w[0].cpu().numpy()}")
    print(f"  Root orientation (wxyz): {robot.data.root_quat_w[0].cpu().numpy()}")
    print(f"  Joint positions: {robot.data.joint_pos[0].cpu().numpy()}")
    print(f"  Joint names: {robot.data.joint_names}")
    print()
    
    # Run simulation loop - just step physics, no actions
    step = 0
    zero_actions = torch.zeros(1, env.num_actions, device=env.device)
    
    try:
        while simulation_app.is_running():
            # Step with zero actions (maintains current joint targets)
            env.step(zero_actions)
            step += 1
            
            # Print state every 50 steps
            if step % 50 == 0:
                pos = robot.data.root_pos_w[0].cpu().numpy()
                vel = robot.data.root_lin_vel_w[0].cpu().numpy()
                joint_pos = robot.data.joint_pos[0].cpu().numpy()
                
                print(f"Step {step}:")
                print(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                print(f"  Velocity: [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")
                print(f"  Joint pos: {joint_pos}")
                print()
            
            # Check for termination
            if env.termination_manager.dones.any():
                term_names = env.termination_manager.active_terms
                for i, name in enumerate(term_names):
                    if env.termination_manager.get_term(name).any():
                        print(f"TERMINATED by: {name}")
                print(f"Episode ended at step {step}")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    print(f"\nTotal steps: {step}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
