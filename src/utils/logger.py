"""
Weights & Biases logging utilities.

This module provides helper functions for integrating W&B logging
into the RL training pipeline.
"""

import wandb
from typing import Dict, Any, Optional


class WandBLogger:
    """
    Wrapper class for Weights & Biases logging.
    
    Provides a simple interface for logging metrics, hyperparameters,
    and other experiment information to W&B.
    """
    
    def __init__(
        self,
        project: str,
        config: Dict[str, Any],
        name: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None
    ):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            config: Configuration dictionary (hyperparameters)
            name: Run name (optional)
            tags: List of tags for the run (optional)
            notes: Notes about the run (optional)
        """
        self.run = wandb.init(
            project=project,
            config=config,
            name=name,
            tags=tags,
            notes=notes
        )
        self.config = config
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for the metrics
        """
        wandb.log(metrics, step=step)
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        duration: int,
        **kwargs
    ) -> None:
        """
        Log episode-specific metrics.
        
        Args:
            episode: Episode number
            reward: Total episode reward
            duration: Episode duration (steps)
            **kwargs: Additional metrics to log
        """
        metrics = {
            'episode': episode,
            'episode_reward': reward,
            'episode_duration': duration,
            **kwargs
        }
        self.log(metrics, step=episode)
    
    def watch_model(self, model, log: str = "gradients", log_freq: int = 100) -> None:
        """
        Watch a PyTorch model for gradient and parameter logging.
        
        Args:
            model: PyTorch model to watch
            log: What to log ("gradients", "parameters", "all", or None)
            log_freq: Logging frequency
        """
        wandb.watch(model, log=log, log_freq=log_freq)
    
    def save_model(self, path: str, name: str = "model") -> None:
        """
        Save a model artifact to W&B.
        
        Args:
            path: Path to the model file
            name: Name for the artifact
        """
        artifact = wandb.Artifact(name, type='model')
        artifact.add_file(path)
        self.run.log_artifact(artifact)
    
    def finish(self) -> None:
        """Finish the W&B run."""
        wandb.finish()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


def init_wandb(
    project: str,
    config: Dict[str, Any],
    name: Optional[str] = None,
    **kwargs
) -> wandb.run:
    """
    Initialize a W&B run.
    
    Args:
        project: W&B project name
        config: Configuration dictionary
        name: Run name (optional)
        **kwargs: Additional arguments for wandb.init()
        
    Returns:
        W&B run object
    """
    return wandb.init(
        project=project,
        config=config,
        name=name,
        **kwargs
    )


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """
    Log metrics to the current W&B run.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number
    """
    wandb.log(metrics, step=step)


def finish_run() -> None:
    """Finish the current W&B run."""
    wandb.finish()
