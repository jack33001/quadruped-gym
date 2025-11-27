"""
Evaluation script for trained quadruped policies.
Loads checkpoint and runs evaluation with optional video recording.
"""
import os
import sys

os.environ["PYTORCH_NVFUSER_DISABLE_FALLBACK"] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

# Configuration - edit these values as needed
HEADLESS = False  # Set to False to see live visualization
RECORD_VIDEO = False  # Video recording in headless mode is complex, disable for now
VIDEO_PATH = "logs/quadruped_walking/eval_video.mp4"
VIDEO_FPS = 30
NUM_ENVS = 16
NUM_EPISODES = 32
EXPERIMENT_NAME = "quadruped_walking"
CHECKPOINT = "model_149.pt"

from isaaclab.app import AppLauncher

# Create app launcher
app_launcher = AppLauncher(
    headless=HEADLESS,
)
simulation_app = app_launcher.app

# Now we can import Isaac Lab modules
import pickle
import torch
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.modules import ActorCritic

from configs import QuadrupedEnvCfg
from quadruped_env import IsaacLabVecEnvWrapper


def evaluate():
    """Evaluate trained policy."""
    
    # Load saved configurations
    log_dir = f"logs/{EXPERIMENT_NAME}"
    if not os.path.exists(log_dir):
        raise ValueError(f"Log directory {log_dir} not found")
    
    checkpoint_path = f"{log_dir}/{CHECKPOINT}"
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint {checkpoint_path} not found")
    
    # Create environment with fewer envs for evaluation
    env_cfg = QuadrupedEnvCfg()
    env_cfg.scene.num_envs = NUM_ENVS
    
    print("\n" + "=" * 80)
    print("QUADRUPED POLICY EVALUATION")
    print("=" * 80)
    print(f"Log directory: {log_dir}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Num environments: {NUM_ENVS}")
    print(f"Num episodes: {NUM_EPISODES}")
    print(f"Headless: {HEADLESS}")
    print("=" * 80 + "\n")
    
    # Create environment
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = IsaacLabVecEnvWrapper(isaac_env)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=env.device, weights_only=False)
    
    # Get initial observation for policy construction
    obs_dict = env.reset()
    
    # Create policy with same architecture as training
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
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        policy.load_state_dict(checkpoint['model'])
    else:
        raise KeyError(f"Checkpoint missing model weights. Keys: {list(checkpoint.keys())}")
    
    policy.eval()
    print("Policy loaded successfully\n")
    
    # Run evaluation
    print("Running evaluation...")
    if not HEADLESS:
        print("Live visualization enabled - watch the simulator window")
        print("Press Ctrl+C to stop\n")
    
    obs_dict = env.reset()
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)
    completed_episodes = 0
    step_count = 0
    
    all_episode_rewards = []
    all_episode_lengths = []
    
    try:
        while completed_episodes < NUM_EPISODES:
            with torch.no_grad():
                actions = policy.act_inference(obs_dict)
            
            obs_dict, rewards, dones, extras = env.step(actions)
            
            episode_rewards += rewards
            episode_lengths += 1
            step_count += 1
            
            # Log progress every 100 steps
            if step_count % 100 == 0 and env.num_envs == 1:
                print(f"  Step {step_count}: reward={rewards[0].item():.3f}, "
                      f"episode_reward={episode_rewards[0].item():.2f}")
            
            # Handle episode terminations
            if dones.any():
                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
                for idx in done_indices:
                    completed_episodes += 1
                    ep_reward = episode_rewards[idx].item()
                    ep_length = episode_lengths[idx].item()
                    
                    all_episode_rewards.append(ep_reward)
                    all_episode_lengths.append(ep_length)
                    
                    print(f"Episode {completed_episodes}: "
                          f"Reward = {ep_reward:.2f}, "
                          f"Length = {ep_length:.0f}")
                    
                    if completed_episodes >= NUM_EPISODES:
                        break
                
                episode_rewards[done_indices] = 0
                episode_lengths[done_indices] = 0
    
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    if len(all_episode_rewards) > 0:
        print(f"Episodes completed: {len(all_episode_rewards)}")
        print(f"Mean reward: {np.mean(all_episode_rewards):.2f} +/- {np.std(all_episode_rewards):.2f}")
        print(f"Mean length: {np.mean(all_episode_lengths):.0f} +/- {np.std(all_episode_lengths):.0f}")
        print(f"Max reward: {np.max(all_episode_rewards):.2f}")
        print(f"Min reward: {np.min(all_episode_rewards):.2f}")
    print("=" * 80)
    
    print("\nEvaluation complete!")
    
    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    evaluate()
