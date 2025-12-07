"""
Main training script for RL algorithms.

This script handles training of PPO, SAC, and TD3 algorithms on Box2D
Gymnasium environments (LunarLander-v3, CarRacing-v3) with Weights & Biases logging.
Supports continuous action spaces for all algorithms.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import gymnasium as gym

# Add src to path
sys.path.append(str(Path(__file__).parent))

from environment import EnvironmentWrapper
from algorithms import PPO, SAC, TD3
from utils.logger import WandBLogger


def load_config(algorithm: str, environment: str) -> dict:
    """
    Load configuration for the specified algorithm and environment.
    
    Args:
        algorithm: Algorithm name (ppo, sac, td3)
        environment: Environment name (LunarLander-v3, CarRacing-v3)
        
    Returns:
        Configuration dictionary for the environment
    """
    config_path = Path(__file__).parent.parent / "configs" / f"{algorithm}_config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    if environment not in configs:
        raise ValueError(
            f"Environment {environment} not found in {algorithm} config. "
            f"Available: {list(configs.keys())}"
        )
    
    return configs[environment]


def get_algorithm_class(algorithm: str):
    """
    Get the algorithm class based on the algorithm name.
    
    Args:
        algorithm: Algorithm name (ppo, sac, td3)
        
    Returns:
        Algorithm class
    """
    algorithms = {
        'ppo': PPO,
        'sac': SAC,
        'td3': TD3
    }
    
    if algorithm not in algorithms:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Available: {list(algorithms.keys())}"
        )
    
    return algorithms[algorithm]


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train RL algorithms on Box2D Gymnasium environments"
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        required=True,
        choices=['ppo', 'sac', 'td3'],
        help='Algorithm to use for training'
    )
    parser.add_argument(
        '--environment',
        type=str,
        required=True,
        choices=['LunarLander-v3', 'CarRacing-v3'],
        help='Box2D Gymnasium environment to train on'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='rl-assignment4',
        help='W&B project name'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training (cpu/cuda)'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render the environment during training'
    )
    parser.add_argument(
        '--record-video',
        action='store_true',
        help='Record videos of episodes'
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        default='videos/',
        help='Directory to save recorded videos'
    )
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    
    # Load configuration
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Environment: {args.environment}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    
    config = load_config(args.algorithm, args.environment)
    print(f"\nHyperparameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    # Initialize environment
    print(f"Initializing environment: {args.environment}...")
    render_mode = 'human' if args.render else None
    
    env = EnvironmentWrapper(
        env_name=args.environment,
        render_mode=render_mode,
        record_video=args.record_video,
        max_steps=config.get('max_episode_steps', 1000),
        video_dir=args.video_dir
    )
    
    print(env)
    print()
    
    # Get state and action dimensions
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action space type: Continuous (Box2D)\n")
    
    # Initialize algorithm
    print(f"Initializing {args.algorithm.upper()} algorithm...")
    AlgorithmClass = get_algorithm_class(args.algorithm)
    agent = AlgorithmClass(state_dim, action_dim, config)
    print(f"{args.algorithm.upper()} agent initialized successfully.\n")
    
    # Initialize W&B logger
    logger = None
    if not args.no_wandb:
        print("Initializing Weights & Biases...")
        logger = WandBLogger(
            project=args.wandb_project,
            config={
                'algorithm': args.algorithm,
                'environment': args.environment,
                'seed': args.seed,
                'device': args.device,
                **config
            },
            name=f"{args.algorithm}_{args.environment}_seed{args.seed}"
        )
        print("W&B logger initialized.\n")
    
    # Create models directory
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    
    # Train the agent
    print(f"{'='*60}")
    print(f"Starting Training")
    print(f"{'='*60}")
    print(f"Training for {config.get('episodes', 1000)} episodes...\n")
    
    try:
        training_stats = agent.train(env, config=config, logger=logger)
        
        # Log final statistics
        print(f"\n{'='*60}")
        print(f"Training Completed!")
        print(f"{'='*60}")
        print(f"Training statistics:")
        for key, value in training_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        print(f"{'='*60}\n")
        
        if logger:
            logger.log({"final_stats": training_stats})
            logger.finish()
        
    except NotImplementedError as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Training method not implemented!")
        print(f"{'='*60}")
        print(f"{e}")
        print(f"\nPlease implement the train() method in algorithms/{args.algorithm}.py")
        print(f"{'='*60}\n")
        
        if logger:
            logger.finish()
        env.close()
        sys.exit(1)
        
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print(f"Training Interrupted by User!")
        print(f"{'='*60}\n")
        
        if logger:
            logger.finish()
        env.close()
        sys.exit(0)
    
    # Save the trained model
    model_path = model_dir / f"{args.algorithm}_{args.environment}_seed{args.seed}.pth"
    
    try:
        agent.save(str(model_path))
        print(f"Model saved to: {model_path}\n")
    except (NotImplementedError, AttributeError) as e:
        print(f"Warning: Could not save model - {e}\n")
    
    # Close environment
    env.close()
    print(f"{'='*60}")
    print(f"Training Session Finished!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
