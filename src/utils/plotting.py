"""
Plotting utilities for visualizing RL training and testing results.

This module provides functions for creating various plots and charts
to analyze agent performance.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path


def plot_episode_durations(
    durations: List[int],
    title: str = "Episode Durations",
    xlabel: str = "Episode",
    ylabel: str = "Duration (steps)",
    save_path: Optional[str] = None,
    show: bool = False,
    window_size: int = 10
) -> None:
    """
    Plot episode durations with optional moving average.
    
    Args:
        durations: List of episode durations
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
        window_size: Window size for moving average
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    episodes = range(1, len(durations) + 1)
    ax.plot(episodes, durations, alpha=0.6, label='Episode Duration')
    
    # Add moving average if enough data points
    if len(durations) >= window_size:
        moving_avg = np.convolve(
            durations,
            np.ones(window_size) / window_size,
            mode='valid'
        )
        avg_episodes = range(window_size, len(durations) + 1)
        ax.plot(avg_episodes, moving_avg, 'r-', linewidth=2, 
                label=f'{window_size}-Episode Moving Average')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_rewards(
    rewards: List[float],
    title: str = "Episode Rewards",
    xlabel: str = "Episode",
    ylabel: str = "Total Reward",
    save_path: Optional[str] = None,
    show: bool = False,
    window_size: int = 10
) -> None:
    """
    Plot episode rewards with optional moving average.
    
    Args:
        rewards: List of episode rewards
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
        window_size: Window size for moving average
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    episodes = range(1, len(rewards) + 1)
    ax.plot(episodes, rewards, alpha=0.6, label='Episode Reward')
    
    # Add moving average if enough data points
    if len(rewards) >= window_size:
        moving_avg = np.convolve(
            rewards,
            np.ones(window_size) / window_size,
            mode='valid'
        )
        avg_episodes = range(window_size, len(rewards) + 1)
        ax.plot(avg_episodes, moving_avg, 'r-', linewidth=2,
                label=f'{window_size}-Episode Moving Average')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_statistics_plot(
    values: List[float],
    title: str = "Statistics",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Create a box plot and histogram for a list of values.
    
    Args:
        values: List of numerical values
        title: Plot title
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    ax1.boxplot(values, vert=True)
    ax1.set_ylabel('Value')
    ax1.set_title(f'{title} - Box Plot')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (
        f'Mean: {np.mean(values):.2f}\n'
        f'Std: {np.std(values):.2f}\n'
        f'Min: {np.min(values):.2f}\n'
        f'Max: {np.max(values):.2f}'
    )
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    # Histogram
    ax2.hist(values, bins=20, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{title} - Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_curves(
    rewards: List[float],
    durations: List[int],
    losses: Optional[List[float]] = None,
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot multiple training curves in subplots.
    
    Args:
        rewards: List of episode rewards
        durations: List of episode durations
        losses: List of loss values (optional)
        title: Overall plot title
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    num_plots = 3 if losses is not None else 2
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))
    
    if num_plots == 2:
        axes = [axes[0], axes[1]]
    else:
        axes = [axes[0], axes[1], axes[2]]
    
    episodes = range(1, len(rewards) + 1)
    
    # Rewards
    axes[0].plot(episodes, rewards, alpha=0.6)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].grid(True, alpha=0.3)
    
    # Durations
    axes[1].plot(episodes, durations, alpha=0.6, color='green')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Duration (steps)')
    axes[1].set_title('Episode Durations')
    axes[1].grid(True, alpha=0.3)
    
    # Losses (if provided)
    if losses is not None:
        loss_steps = range(1, len(losses) + 1)
        axes[2].plot(loss_steps, losses, alpha=0.6, color='red')
        axes[2].set_xlabel('Update Step')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Training Loss')
        axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    data_dict: dict,
    title: str = "Algorithm Comparison",
    xlabel: str = "Episode",
    ylabel: str = "Value",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot comparison of multiple algorithms or runs.
    
    Args:
        data_dict: Dictionary mapping labels to data lists
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for label, data in data_dict.items():
        episodes = range(1, len(data) + 1)
        ax.plot(episodes, data, alpha=0.7, label=label, linewidth=2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
