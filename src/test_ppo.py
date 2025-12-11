
"""
Testing script for trained RL agents.

This script loads trained models and evaluates them on test episodes,
collecting statistics about performance for Box2D environments.
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import csv
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent))

from environment_ppo import EnvironmentWrapper
from algorithms import TD3, SAC, PPO_LunarLander, PPO_CarRacing
from algorithms.ppo_carracing import PPO_CarRacing   # <-- NEW
from utils.plotting import save_statistics_plot


def load_config(algorithm: str, environment: str) -> dict:
    """
    Load configuration for the specified algorithm and environment.
    
    Args:
        algorithm: Algorithm name (td3, sac, ppo)
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


def get_algorithm_class(algorithm: str, environment: str):
    """
    Get the algorithm class based on the algorithm name + environment.
    
    Args:
        algorithm: Algorithm name (td3, sac, ppo)
        environment: Environment name (LunarLander-v3, CarRacing-v3)
        
    Returns:
        Algorithm class
    """
    if algorithm == "ppo":
        # Use image-based NatureCNN PPO for CarRacing
        if environment == "CarRacing-v3":
            return PPO_CarRacing
        # Use standard MLP PPO for LunarLander
        return PPO_LunarLander

    algorithms = {
        "td3": TD3,
        "sac": SAC,
    }
    
    if algorithm not in algorithms:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Available: ['td3', 'sac', 'ppo']"
        )
    
    return algorithms[algorithm]


def compute_statistics(values: list) -> Dict[str, float]:
    """
    Compute statistics from a list of values.
    
    Args:
        values: List of numerical values
        
    Returns:
        Dictionary with mean, std, min, max
    """
    values_array = np.array(values)
    return {
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array))
    }


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(
        description="Test trained RL agents on Box2D Gymnasium environments"
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        required=True,
        choices=['td3', 'sac', 'ppo'],
        help='Algorithm used for training'
    )
    parser.add_argument(
        '--environment',
        type=str,
        required=True,
        choices=['LunarLander-v3', 'CarRacing-v3'],
        help='Box2D Gymnasium environment to test on'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to the trained model (default: models/{algo_variant}_{environment}_seed42.pth)'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=100,
        help='Number of test episodes (default: 100)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render the environment during testing'
    )
    parser.add_argument(
        '--record-video',
        action='store_true',
        help='Record videos of test episodes'
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        default='test_videos/',
        help='Directory to save recorded videos'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save test results and plots to results/ directory'
    )
    
    args = parser.parse_args()
    
    # Variant naming like train.py: ppo_lunarlander / ppo_carracing / sac / td3
    if args.algorithm == "ppo":
        if args.environment == "LunarLander-v3":
            algo_variant = "ppo_lunarlander"
        elif args.environment == "CarRacing-v3":
            algo_variant = "ppo_carracing"
        else:
            algo_variant = "ppo"
    else:
        algo_variant = args.algorithm
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine model path
    if args.model_path is None:
        model_path = (
            Path(__file__).parent.parent
            / "models"
            / f"{algo_variant}_{args.environment}_seed{args.seed}.pth"
        )
    else:
        model_path = Path(args.model_path)
    
    if not model_path.exists():
        print(f"\n{'='*60}")
        print(f"ERROR: Model file not found!")
        print(f"{'='*60}")
        print(f"Looking for: {model_path}")
        print(f"\nPlease train a model first using:")
        print(f"  python src/train.py --algorithm {args.algorithm} --environment {args.environment} --seed {args.seed}")
        print(f"{'='*60}\n")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Testing Configuration")
    print(f"{'='*60}")
    print(f"Algorithm: {algo_variant.upper()}")
    print(f"Environment: {args.environment}")
    print(f"Model path: {model_path}")
    print(f"Number of test episodes: {args.num_episodes}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")
    
    # Load configuration (still keyed by base algorithm name)
    config = load_config(args.algorithm, args.environment)
    
    # Decide whether to use CarRacing CNN wrapper (same logic as train.py)
    if args.environment == "CarRacing-v3" and args.algorithm == "ppo":
        use_carracing_cnn = False   # PPO_CarRacing uses raw images
    else:
        use_carracing_cnn = True    # SAC/TD3 use CNN wrapper
    
    # Initialize environment
    render_mode = 'human' if args.render else None
    print(f"Initializing environment: {args.environment}...")
    
    env = EnvironmentWrapper(
        env_name=args.environment,
        render_mode=render_mode,
        record_video=args.record_video,
        max_steps=config.get('max_episode_steps', 1000),
        video_dir=args.video_dir,
        use_carracing_cnn=use_carracing_cnn,
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
    print(f"Initializing {algo_variant.upper()} algorithm...")
    AlgorithmClass = get_algorithm_class(args.algorithm, args.environment)
    agent = AlgorithmClass(state_dim, action_dim, config)
    
    # Load trained model
    try:
        agent.load(str(model_path))
        print(f"{algo_variant.upper()} model loaded successfully!\n")
    except (NotImplementedError, AttributeError) as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Failed to load model!")
        print(f"{'='*60}")
        print(f"{e}")
        print(f"Please ensure the load() method is implemented correctly.")
        print(f"{'='*60}\n")
        env.close()
        sys.exit(1)
    
    # Run test episodes
    print(f"{'='*60}")
    print(f"Running Test Episodes")
    print(f"{'='*60}\n")
    
    try:
        # Prefer evaluate(), fallback to test()
        if hasattr(agent, 'evaluate'):
            test_stats = agent.evaluate(env, num_episodes=args.num_episodes)
        elif hasattr(agent, 'test'):
            test_stats = agent.test(env, num_episodes=args.num_episodes)
        else:
            raise NotImplementedError(
                f"Neither evaluate() nor test() method found in {AlgorithmClass.__name__}"
            )
        
        # Display statistics
        print(f"\n{'='*60}")
        print(f"TEST RESULTS")
        print(f"{'='*60}")
        for key, value in test_stats.items():
            if isinstance(value, (list, np.ndarray)):
                # Don't print full arrays
                continue
            elif isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print(f"{'='*60}\n")
        
        # Save results if requested
        if args.save_results:
            results_dir = Path(__file__).parent.parent / "results"
            results_dir.mkdir(exist_ok=True)
            
            # Save statistics to text file
            results_file = results_dir / f"{algo_variant}_{args.environment}_test_results.txt"
            with open(results_file, 'w') as f:
                f.write(f"Test Results for {algo_variant.upper()} on {args.environment}\n")
                f.write(f"Model: {model_path}\n")
                f.write(f"Number of episodes: {args.num_episodes}\n")
                f.write(f"Seed: {args.seed}\n")
                f.write("="*60 + "\n\n")
                
                for key, value in test_stats.items():
                    if key not in ['episode_rewards', 'episode_durations']:
                        if isinstance(value, float):
                            f.write(f"{key}: {value:.4f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
            
            print(f"Results saved to: {results_file}")
            
            # Save detailed episode data to CSV if available
            if 'episode_rewards' in test_stats:
                csv_file = results_dir / f"{algo_variant}_{args.environment}_test_episodes.csv"
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    has_durations = 'episode_durations' in test_stats
                    
                    if has_durations:
                        writer.writerow(['Episode', 'Reward', 'Duration'])
                        for i, (reward, duration) in enumerate(
                            zip(test_stats['episode_rewards'], test_stats['episode_durations']), 1
                        ):
                            writer.writerow([i, reward, duration])
                    else:
                        writer.writerow(['Episode', 'Reward'])
                        for i, reward in enumerate(test_stats['episode_rewards'], 1):
                            writer.writerow([i, reward])
                
                print(f"Episode data saved to: {csv_file}")
            
            # Generate and save plots
            if 'episode_rewards' in test_stats:
                plot_path = results_dir / f"{algo_variant}_{args.environment}_test_rewards.png"
                save_statistics_plot(
                    test_stats['episode_rewards'],
                    title=f"{algo_variant.upper()} on {args.environment} - Test Episode Rewards",
                    ylabel="Episode Reward",
                    save_path=str(plot_path)
                )
                print(f"Rewards plot saved to: {plot_path}")
            
            if 'episode_durations' in test_stats:
                plot_path = results_dir / f"{algo_variant}_{args.environment}_test_durations.png"
                save_statistics_plot(
                    test_stats['episode_durations'],
                    title=f"{algo_variant.upper()} on {args.environment} - Test Episode Durations",
                    ylabel="Episode Duration (steps)",
                    save_path=str(plot_path)
                )
                print(f"Durations plot saved to: {plot_path}")
            
            print()
        
    except NotImplementedError as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Testing method not implemented!")
        print(f"{'='*60}")
        print(f"{e}")
        print(f"{'='*60}\n")
        env.close()
        sys.exit(1)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR during testing!")
        print(f"{'='*60}")
        print(f"{e}")
        print(f"{'='*60}\n")
        env.close()
        sys.exit(1)
    
    # Close environment
    env.close()
    print(f"{'='*60}")
    print(f"Testing Session Finished!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
