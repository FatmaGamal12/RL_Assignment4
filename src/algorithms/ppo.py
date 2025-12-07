"""
Proximal Policy Optimization (PPO) Algorithm Implementation.

PPO is an on-policy actor-critic algorithm that uses a clipped surrogate
objective to limit policy updates and improve training stability.

Reference: https://arxiv.org/abs/1707.06347
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List


# ============================
#  ACTOR NETWORK (Policy)
# ============================
class Actor(nn.Module):
    """
    Policy network for continuous action spaces.
    Outputs mean and log_std for a Gaussian policy.
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Mean of the action distribution
        self.mean = nn.Linear(256, action_dim)
        
        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the actor network.
        
        Args:
            state: State tensor
            
        Returns:
            mean: Mean of the action distribution
            std: Standard deviation of the action distribution
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = torch.tanh(self.mean(x))  # Bound actions to [-1, 1]
        std = self.log_std.exp().expand_as(mean)
        
        return mean, std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy distribution.
        
        Args:
            state: State tensor
            
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
        """
        mean, std = self(state)
        
        # Create normal distribution
        dist = torch.distributions.Normal(mean, std)
        
        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under the current policy.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            log_prob: Log probability of the actions
            entropy: Entropy of the policy distribution
            mean: Mean of the action distribution
        """
        mean, std = self(state)
        
        # Create normal distribution
        dist = torch.distributions.Normal(mean, std)
        
        # Calculate log probability and entropy
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy, mean


# ============================
#  CRITIC NETWORK (Value Function)
# ============================
class Critic(nn.Module):
    """
    Value network that estimates the state value function V(s).
    """
    
    def __init__(self, state_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.value = nn.Linear(256, 1)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network.
        
        Args:
            state: State tensor
            
        Returns:
            value: Estimated state value
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        
        return value


# ============================
#  PPO MAIN CLASS
# ============================
class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm with clipped surrogate objective.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Initialize PPO agent.
        
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
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.batch_size = config.get("batch_size", 64)
        self.n_steps = config.get("n_steps", 2048)
        self.n_epochs = config.get("n_epochs", 10)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # Storage for trajectories
        self.reset_storage()
    
    def reset_storage(self):
        """Reset trajectory storage buffers."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select an action using the current policy.
        
        Args:
            state: Current state
            deterministic: If True, return the mean action (for evaluation)
            
        Returns:
            action: Selected action
        """
        # TODO: Implement action selection
        # 1. Convert state to tensor
        # 2. Get action from actor network
        # 3. If deterministic, return mean; otherwise sample from distribution
        # 4. Store action, log_prob, and value for training
        # 5. Return action as numpy array
        
        raise NotImplementedError("select_action method needs to be implemented")
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool], 
                    next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of state values
            dones: List of done flags
            next_value: Value of the next state
            
        Returns:
            advantages: Computed advantages
            returns: Computed returns (targets for value function)
        """
        # TODO: Implement GAE computation
        # 1. Initialize lists for advantages and returns
        # 2. Calculate TD errors (delta = r + gamma * V(s') - V(s))
        # 3. Compute advantages using GAE formula
        # 4. Compute returns (advantages + values)
        # 5. Return advantages and returns as tensors
        
        raise NotImplementedError("compute_gae method needs to be implemented")
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update policy and value networks using collected trajectories.
        
        Returns:
            losses: Dictionary containing loss values
        """
        # TODO: Implement policy update
        # 1. Compute GAE advantages and returns
        # 2. Normalize advantages
        # 3. Convert stored data to tensors
        # 4. For each epoch:
        #    a. Create mini-batches
        #    b. For each mini-batch:
        #       - Evaluate actions under current policy
        #       - Compute policy loss (clipped surrogate objective)
        #       - Compute value loss
        #       - Compute entropy bonus
        #       - Update networks
        # 5. Reset storage
        # 6. Return average losses
        
        raise NotImplementedError("update_policy method needs to be implemented")
    
    def train(self, env, config: Dict[str, Any] = None, logger=None) -> Dict[str, Any]:
        """
        Train the PPO agent on the given environment.
        
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
        # 3. Main training loop:
        #    a. Reset environment
        #    b. Collect trajectories (n_steps)
        #    c. Update policy using collected data
        #    d. Log metrics to W&B (if logger provided)
        #    e. Track episode statistics
        # 4. Return training statistics (mean/max/min rewards, etc.)
        
        raise NotImplementedError(
            "Training method for PPO needs to be implemented. "
            "Please implement the train() method in algorithms/ppo.py"
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
        # 1. Set policy to evaluation mode
        # 2. Run episodes using deterministic actions
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
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """
        Load the model parameters.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
