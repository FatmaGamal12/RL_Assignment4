"""
Main training script for RL algorithms.

This script handles training of PPO (LunarLander + CarRacing),
SAC, and TD3 on Box2D Gymnasium environments, with W&B logging.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import gymnasium as gym

# Add src/ to Python path
sys.path.append(str(Path(__file__).parent))

from environment import EnvironmentWrapper
from algorithms import (
    PPO_LunarLander,
    PPO_CarRacing,
    SAC,
    TD3
)
from utils.logger import WandBLogger


# ------------------------------------------------------------------
# CONFIG LOADER
# ------------------------------------------------------------------
def load_config(algorithm: str, environment: str) -> dict:
    """
    Load YAML config for chosen algorithm & env.
    """
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


# ------------------------------------------------------------------
# ALGORITHM RESOLVER
# ------------------------------------------------------------------
def resolve_algorithm(algorithm: str, environment: str):
    """
    Decide which class to use:
    PPO_LunarLander  (vector input)
    PPO_CarRacing    (raw pixels + CNN)
    SAC              (vector/CNN based on wrapper)
    TD3
    """
    if algorithm == "ppo":
        if environment == "LunarLander-v3":
            return PPO_LunarLander
        elif environment == "CarRacing-v3":
            return PPO_CarRacing
        else:
            raise ValueError(f"PPO does not support env: {environment}")

    elif algorithm == "sac":
        return SAC
    elif algorithm == "td3":
        return TD3
    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Allowed: ppo, sac, td3"
        )


# ------------------------------------------------------------------
# TRAINING ENTRY POINT
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train RL algorithms.")
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['ppo', 'sac', 'td3'])
    parser.add_argument('--environment', type=str, required=True,
                        choices=['LunarLander-v3', 'CarRacing-v3'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='rl-assignment4')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--record-video', action='store_true')
    parser.add_argument('--video-dir', type=str, default='videos/')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve algorithm variant name for logging
    # ------------------------------------------------------------------
    if args.algorithm == "ppo":
        if args.environment == "LunarLander-v3":
            algo_variant = "ppo_lunarlander"
        else:
            algo_variant = "ppo_carracing"
    else:
        algo_variant = args.algorithm

    # ------------------------------------------------------------------
    # Set seeds
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Training Configuration")
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

    # ------------------------------------------------------------------
    # Choose whether to use shared CarRacing CNN wrapper
    # PPO gets raw pixels → CNN inside PPO_CarRacing
    # SAC/TD3 get 128-dim embedding from wrapper
    # ------------------------------------------------------------------
    use_carracing_cnn = not (args.algorithm == "ppo" and args.environment == "CarRacing-v3")

    # ------------------------------------------------------------------
    # Initialize environment
    # ------------------------------------------------------------------
    print(f"Initializing environment: {args.environment}...")
    render_mode = "human" if args.render else None

    env = EnvironmentWrapper(
        env_name=args.environment,
        render_mode=render_mode,
        record_video=args.record_video,
        max_steps=config.get("max_episode_steps", 1000),
        video_dir=args.video_dir,
        use_carracing_cnn=use_carracing_cnn
    )

    print(env)
    print()

    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action space: Continuous\n")

    # ------------------------------------------------------------------
    # Initialize algorithm class
    # ------------------------------------------------------------------
    AlgoClass = resolve_algorithm(args.algorithm, args.environment)

    print(f"Initializing {algo_variant.upper()} agent...")
    agent = AlgoClass(state_dim, action_dim, config)
    print("Agent initialized successfully.\n")

    # ------------------------------------------------------------------
    # Initialize W&B logger
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    print(f"{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")

    try:
        training_stats = agent.train(env, config=config, logger=logger)

        # Print summary
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
        env.close()
        sys.exit(0)

    # ------------------------------------------------------------------
    # SAVE FINAL MODEL
    # ------------------------------------------------------------------
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / f"{algo_variant}_{args.environment}_seed{args.seed}.pth"

    try:
        agent.save(str(model_path))
        print(f"Model saved to: {model_path}")
    except Exception as e:
        print("WARNING: Could not save model:", e)

    env.close()
    print("\nTraining Session Finished!\n")


if __name__ == "__main__":
    main()
