"""
Evaluation script for trained quadruped policies.
Loads checkpoint and visualizes policy.
"""
import argparse
import pickle
import os

import torch
import genesis as gs
from rsl_rl.modules import ActorCritic

from quadruped_env import QuadrupedEnv
from configs import EvalConfig


def evaluate(args):
    """Evaluate trained policy."""
    
    # Load configuration
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        raise ValueError(f"Log directory {log_dir} not found")
    
    cfgs = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    env_cfg = cfgs['env_cfg']
    obs_cfg = cfgs['obs_cfg']
    reward_cfg = cfgs['reward_cfg']
    command_cfg = cfgs['command_cfg']
    train_cfg = cfgs['train_cfg']
    
    # Backward compatibility: use default EvalConfig if not in saved configs
    eval_cfg = cfgs.get('eval_cfg', EvalConfig())
    
    # Override settings for evaluation from EvalConfig
    env_cfg.num_envs = eval_cfg.num_envs
    env_cfg.show_viewer = True  # Always show viewer for eval
    
    # Initialize Genesis
    gs.init(logging_level="warning")
    
    # Create environment
    print("\n" + "="*80)
    print("QUADRUPED POLICY EVALUATION")
    print("="*80)
    print(f"Log directory: {log_dir}")
    print(f"Checkpoint: {eval_cfg.checkpoint}")
    print(f"Num environments: {eval_cfg.num_envs}")
    print(f"Num episodes: {eval_cfg.num_episodes}")
    print("="*80 + "\n")
    
    env = QuadrupedEnv(
        num_envs=env_cfg.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )
    
    # Load policy
    checkpoint_path = f"{log_dir}/{eval_cfg.checkpoint}"
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint {checkpoint_path} not found")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=env.device)
    
    # Debug: print available keys
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Get initial observation for policy construction
    obs_dict = env.reset()
    obs_groups = {
        'policy': ['policy'],
        'critic': ['policy'],
    }
    
    # Create policy
    policy = ActorCritic(
        obs=obs_dict,
        obs_groups=obs_groups,
        num_actions=env.num_actions,
        **train_cfg.policy
    ).to(env.device)
    
    # Load weights - RSL-RL uses 'model_state_dict' key
    if 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        policy.load_state_dict(checkpoint['model'])
    else:
        raise KeyError(f"Checkpoint does not contain 'model_state_dict' or 'model' key. "
                      f"Available keys: {list(checkpoint.keys())}")
    
    policy.eval()
    
    print("Policy loaded successfully\n")
    print("Running evaluation...")
    print("Press Ctrl+C to stop\n")
    
    # Run evaluation
    obs_dict = env.reset()
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)
    num_episodes = 0
    
    # Add step counter for detailed logging
    step_count = 0
    
    try:
        while num_episodes < eval_cfg.num_episodes:
            with torch.no_grad():
                actions = policy.act_inference(obs_dict)
            
            obs_dict, rewards, dones, _ = env.step(actions)
            
            episode_rewards += rewards
            episode_lengths += 1
            step_count += 1
            
            # Log every 50 steps to see what's happening
            if step_count % 50 == 0:
                base_pos = env.robot.get_pos()
                base_vel = env.robot.get_vel()
                base_ang_vel = env.robot.get_ang()
                cmd = env.commands[0]  # Get commanded velocity
                print(f"  Step {step_count}: "
                      f"pos=[{base_pos[0, 0]:.2f}, {base_pos[0, 1]:.2f}, {base_pos[0, 2]:.3f}], "
                      f"vel=[{base_vel[0, 0]:.2f}, {base_vel[0, 1]:.2f}, {base_vel[0, 2]:.2f}], "
                      f"cmd=[vx:{cmd[0]:.2f}, vy:{cmd[1]:.2f}, ω:{cmd[2]:.2f}], "
                      f"ang_vel={base_ang_vel[0, 2]:.2f}, "
                      f"reward={rewards[0].item():.2f}")
            
            # Handle episode terminations
            if dones.any():
                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
                for idx in done_indices:
                    num_episodes += 1
                    print(f"Episode {num_episodes}: "
                          f"Reward = {episode_rewards[idx].item():.2f}, "
                          f"Length = {episode_lengths[idx].item():.0f}")
                    
                    if num_episodes >= eval_cfg.num_episodes:
                        break
                
                episode_rewards[done_indices] = 0
                episode_lengths[done_indices] = 0
    
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    
    print("\nEvaluation complete!")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained quadruped policy")
    
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/quadruped_walking",  # Changed from quadruped_locomotion
        help="Path to log directory containing saved configs and checkpoints"
    )
    
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
