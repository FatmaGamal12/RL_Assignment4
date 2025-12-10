"""
Main training script for RL algorithms - FIXED VERSION

Key changes:
- Support for vectorized environments (multiple parallel envs)
- Proper integration with fixed PPO_CarRacing
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import gymnasium as gym

sys.path.append(str(Path(__file__).parent))

from environment import EnvironmentWrapper
from algorithms import (
    PPO_LunarLander,
    PPO_CarRacing,  # Import the FIXED version
    SAC,
    TD3
)
from utils.logger import WandBLogger


def load_config(algorithm: str, environment: str) -> dict:
    """Load YAML config for chosen algorithm & env."""
    config_path = Path(__file__).parent.parent / "configs" / f"{algorithm}_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    if environment not in configs:
        raise ValueError(
            f"Environment {environment} not found in {algorithm}_config.yaml\n"
            f"Available: {list(configs.keys())}"
        )
    return configs[environment]


def make_vectorized_envs(env_name: str, n_envs: int, config: dict, 
                         render_mode=None, record_video=False):
    """Create vectorized environments for PPO CarRacing."""
    envs = []
    for i in range(n_envs):
        # Only render/record for first env
        current_render = render_mode if i == 0 else None
        current_record = record_video if i == 0 else False
        
        env = EnvironmentWrapper(
            env_name=env_name,
            render_mode=current_render,
            record_video=current_record,
            max_steps=config.get("max_episode_steps", 1000),
            video_dir=f"videos/env_{i}/",
            use_carracing_cnn=False  # PPO uses raw pixels
        )
        envs.append(env.env)  # Get underlying gym env
    return envs


def resolve_algorithm(algorithm: str, environment: str):
    """Decide which class to use."""
    if algorithm == "ppo":
        if environment == "LunarLander-v3":
            return PPO_LunarLander
        elif environment == "CarRacing-v3":
            return PPO_CarRacing  # FIXED version
        else:
            raise ValueError(f"PPO does not support env: {environment}")
    elif algorithm == "sac":
        return SAC
    elif algorithm == "td3":
        return TD3
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'")


def main():
    parser = argparse.ArgumentParser(description="Train RL algorithms (FIXED).")
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['ppo', 'sac', 'td3'])
    parser.add_argument('--environment', type=str, required=True,
                        choices=['LunarLander-v3', 'CarRacing-v3'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='rl-assignment4-fixed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--record-video', action='store_true')
    parser.add_argument('--video-dir', type=str, default='videos/')
    args = parser.parse_args()

    # Resolve algorithm variant
    if args.algorithm == "ppo":
        if args.environment == "LunarLander-v3":
            algo_variant = "ppo_lunarlander"
        else:
            algo_variant = "ppo_carracing_fixed"
    else:
        algo_variant = args.algorithm

    # Set seeds
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    # Load config
    print(f"\n{'='*60}")
    print("Training Configuration (FIXED)")
    print(f"{'='*60}")
    print(f"Algorithm: {algo_variant.upper()}")
    print(f"Environment: {args.environment}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")

    config = load_config(args.algorithm, args.environment)

    print("\nHyperparameters:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    # ============================================================
    # HANDLE VECTORIZED ENVIRONMENTS FOR PPO CARRACING
    # ============================================================
    if args.algorithm == "ppo" and args.environment == "CarRacing-v3":
        n_envs = config.get("n_envs", 8)
        print(f"Creating {n_envs} parallel CarRacing environments...")
        
        render_mode = "human" if args.render else None
        envs = make_vectorized_envs(
            env_name=args.environment,
            n_envs=n_envs,
            config=config,
            render_mode=render_mode,
            record_video=args.record_video
        )
        
        print(f"Created {len(envs)} parallel environments")
        
        # Get dimensions from first env
        state_dim = envs[0].observation_space.shape[0] if len(envs[0].observation_space.shape) == 1 else \
                    int(np.prod(envs[0].observation_space.shape))
        action_dim = envs[0].action_space.shape[0]
        
        print(f"Observation space: {envs[0].observation_space}")
        print(f"Action space: {envs[0].action_space}")
        
    else:
        # Single environment for other algorithms
        use_carracing_cnn = not (args.algorithm == "ppo" and args.environment == "CarRacing-v3")
        render_mode = "human" if args.render else None
        
        env_wrapper = EnvironmentWrapper(
            env_name=args.environment,
            render_mode=render_mode,
            record_video=args.record_video,
            max_steps=config.get("max_episode_steps", 1000),
            video_dir=args.video_dir,
            use_carracing_cnn=use_carracing_cnn
        )
        
        envs = env_wrapper.env  # Single env
        state_dim = env_wrapper.get_state_dim()
        action_dim = env_wrapper.get_action_dim()
        
        print(env_wrapper)

    print(f"\nState dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action space: Continuous\n")

    # ============================================================
    # INITIALIZE ALGORITHM
    # ============================================================
    AlgoClass = resolve_algorithm(args.algorithm, args.environment)

    print(f"Initializing {algo_variant.upper()} agent...")
    agent = AlgoClass(state_dim, action_dim, config)
    print("Agent initialized successfully.\n")

    # Load checkpoint if continuing training
    if args.algorithm == "ppo" and args.environment == "CarRacing-v3":
        ckpt = config.get("checkpoint_path", "models/ppo_CarRacing-v3_best.pth")
        if os.path.exists(ckpt):
            print(f"Found existing checkpoint: {ckpt}")
            try:
                agent.load(ckpt)
                print("✓ Checkpoint loaded successfully")
            except Exception as e:
                print(f"✗ Could not load checkpoint: {e}")
                print("Starting training from scratch...")
        else:
            print("No checkpoint found, training from scratch.")

    # ============================================================
    # INITIALIZE WANDB LOGGER
    # ============================================================
    logger = None
    if not args.no_wandb:
        print("Initializing Weights & Biases...")
        logger = WandBLogger(
            project=args.wandb_project,
            name=f"{algo_variant}_{args.environment}_seed{args.seed}",
            config={
                "algorithm": algo_variant,
                "environment": args.environment,
                "seed": args.seed,
                "device": args.device,
                **config
            }
        )
        print("W&B logger initialized.\n")

    # ============================================================
    # TRAIN
    # ============================================================
    print(f"{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")

    try:
        training_stats = agent.train(envs, config=config, logger=logger)

        print(f"\n{'='*60}")
        print("Training Completed!")
        print(f"{'='*60}")
        for k, v in training_stats.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}\n")

        if logger:
            logger.log({"final_stats": training_stats})
            logger.finish()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        if logger:
            logger.finish()
        
        # Close environments
        if isinstance(envs, list):
            for env in envs:
                env.close()
        else:
            envs.close()
        sys.exit(0)

    # ============================================================
    # SAVE FINAL MODEL
    # ============================================================
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / f"{algo_variant}_{args.environment}_seed{args.seed}.pth"

    try:
        agent.save(str(model_path))
        print(f"Model saved to: {model_path}")
    except Exception as e:
        print("WARNING: Could not save model:", e)

    # Close environments
    if isinstance(envs, list):
        for env in envs:
            env.close()
    else:
        envs.close()
    
    print("\nTraining Session Finished!\n")


if __name__ == "__main__":
    main()