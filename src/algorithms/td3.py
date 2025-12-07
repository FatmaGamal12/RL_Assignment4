"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) Algorithm Implementation.

TD3 is an off-policy actor-critic algorithm that extends DDPG with:
1. Twin Q-networks (clipped double Q-learning)
2. Delayed policy updates
3. Target policy smoothing

Reference: https://arxiv.org/abs/1802.09477
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple
import random
from collections import deque


# ============================
#  ACTOR NETWORK (Deterministic Policy)
# ============================
class Actor(nn.Module):
    """
    Deterministic policy network for continuous action spaces.
    Outputs actions in the range [-1, 1] using tanh activation.
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.action = nn.Linear(256, action_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the actor network.
        
        Args:
            state: State tensor
            
        Returns:
            action: Deterministic action in range [-1, 1]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.action(x))
        
        return action


# ============================
#  CRITIC NETWORK (Q-Function)
# ============================
class Critic(nn.Module):
    """
    Q-network that estimates the action-value function Q(s, a).
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q_value = nn.Linear(256, 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            q_value: Estimated Q-value
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_value(x)
        
        return q_value


# ============================
#  TD3 MAIN CLASS
# ============================
class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Initialize TD3 agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Configuration dictionary containing hyperparameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Hyperparameters
        self.lr = config.get("learning_rate", 3e-4)
        self.gamma = config.get("discount_factor", 0.99)
        self.tau = config.get("tau", 0.005)
        self.policy_noise = config.get("policy_noise", 0.2)
        self.noise_clip = config.get("noise_clip", 0.5)
        self.policy_delay = config.get("policy_delay", 2)
        self.exploration_noise = config.get("exploration_noise", 0.1)
        self.batch_size = config.get("batch_size", 256)
        self.replay_memory_size = config.get("replay_memory_size", 1000000)
        self.warmup_steps = config.get("warmup_steps", 1000)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor networks (policy and target)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Twin Critic networks (Q1, Q2 and their targets)
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.lr)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=self.replay_memory_size)
        
        # Training step counter
        self.total_steps = 0
    
    def select_action(self, state: np.ndarray, deterministic: bool = False, 
                     env_name: str = None) -> np.ndarray:
        """
        Select an action using the current policy.
        
        Args:
            state: Current state
            deterministic: If True, return action without exploration noise
            env_name: Environment name (for action scaling if needed)
            
        Returns:
            action: Selected action
        """
        # TODO: Implement action selection
        # 1. Convert state to tensor
        # 2. Get action from actor network
        # 3. If not deterministic, add exploration noise
        # 4. Clip action to valid range [-1, 1]
        # 5. Handle special action scaling for CarRacing-v3 if needed
        # 6. Return action as numpy array
        
        raise NotImplementedError("select_action method needs to be implemented")
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                         next_state: np.ndarray, done: bool):
        """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Done flag
        """
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def sample_batch(self) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch from the replay buffer.
        
        Returns:
            batch: Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        # TODO: Implement batch sampling
        # 1. Sample random transitions from replay buffer
        # 2. Convert to tensors on the correct device
        # 3. Return tuple of tensors
        
        raise NotImplementedError("sample_batch method needs to be implemented")
    
    def soft_update(self, target: nn.Module, source: nn.Module):
        """
        Soft update of target network parameters.
        θ_target = τ * θ_source + (1 - τ) * θ_target
        
        Args:
            target: Target network
            source: Source network
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def update_networks(self) -> Dict[str, float]:
        """
        Update critic and actor networks using a batch from the replay buffer.
        
        Returns:
            losses: Dictionary containing loss values
        """
        # TODO: Implement network update
        # 1. Sample a batch from replay buffer
        # 2. Compute target Q-values using target networks
        #    - Add clipped noise to target actions (target policy smoothing)
        #    - Use minimum of twin Q-values (clipped double Q-learning)
        # 3. Update critics to minimize TD error
        # 4. Every policy_delay steps:
        #    - Update actor by maximizing Q1(s, actor(s))
        #    - Soft update all target networks
        # 5. Return losses for logging
        
        raise NotImplementedError("update_networks method needs to be implemented")
    
    def train(self, env, config: Dict[str, Any] = None, logger=None) -> Dict[str, Any]:
        """
        Train the TD3 agent on the given environment.
        
        Args:
            env: Gymnasium environment
            config: Configuration dictionary (optional, uses self.config if None)
            logger: Weights & Biases logger (optional)
            
        Returns:
            training_stats: Dictionary containing training statistics
        """
        # TODO: Implement main training loop
        # 1. Get training parameters from config
        # 2. Initialize tracking variables (episode rewards, steps, etc.)
        # 3. Warmup phase: collect random transitions
        # 4. Main training loop:
        #    a. Reset environment for each episode
        #    b. For each step in episode:
        #       - Select action (with exploration noise)
        #       - Execute action in environment
        #       - Store transition in replay buffer
        #       - Update networks if buffer has enough samples
        #       - Track rewards and steps
        #    c. Log metrics to W&B (if logger provided)
        #    d. Track episode statistics
        # 5. Return training statistics (mean/max/min rewards, etc.)
        
        raise NotImplementedError(
            "Training method for TD3 needs to be implemented. "
            "Please implement the train() method in algorithms/td3.py"
        )
    
    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the trained policy.
        
        Args:
            env: Gymnasium environment
            num_episodes: Number of episodes to evaluate
            
        Returns:
            eval_stats: Dictionary containing evaluation statistics
        """
        # TODO: Implement evaluation
        # 1. Set policy to evaluation mode (deterministic actions)
        # 2. Run episodes without exploration noise
        # 3. Track episode rewards
        # 4. Return mean and std of rewards
        
        raise NotImplementedError("evaluate method needs to be implemented")
    
    def save(self, path: str):
        """
        Save the model parameters.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """
        Load the model parameters.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        
        # Also load target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
