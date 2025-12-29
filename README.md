# Quadruped Gym

GPU-accelerated quadruped locomotion training using Isaac Lab and RSL-RL.

## Requirements

- NVIDIA Isaac Lab (with Isaac Sim)
- RSL-RL
- Python 3.10+

## Training

To train, run the training script:

```bash
python train.py
```

Training configuration is defined in `train_cfg.py`. Key parameters:

- `max_iterations`: Number of training iterations (default: 5000)
- `num_envs`: Number of parallel environments (default: 8192)
- `experiment_name`: Name for logging directory

*Currently, the system trains on flat ground.*

Logs and checkpoints are saved to `logs/<experiment_name>/`.

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs
```

## Evaluation

### Flat Ground

Evaluate on flat terrain:

```bash
python evaluate_flat.py
```

### Rough Terrain

Evaluate on varied terrain:

```bash
python evaluate_rough.py
```

Evaluation settings are configured in `train_cfg.py` under `EvalCfg`:

- `headless`: Run without visualization (this will disable video recording)
- `record_video`: Save evaluation video
- `evaluation_duration`: Length of evaluation in seconds

Videos are saved to `logs/<experiment_name>/videos/`.

## Project Structure

- `train.py` - Main training script
- `train_cfg.py` - Training and evaluation configuration
- `rl_cfg.py` - RL environment configuration (observations, rewards, terminations)
- `sim_cfg.py` - Simulation configuration (robot, terrain, sensors)
- `gait_cfg.py` - Gait scheduler configuration
- `quadruped_env.py` - RSL-RL compatible environment wrapper
- `evaluate_base.py` - Base evaluation utilities
- `evaluate_flat.py` - Flat ground evaluation
- `evaluate_rough.py` - Rough terrain evaluation
