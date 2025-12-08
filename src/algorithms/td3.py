"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) Algorithm Implementation.

TD3 is an off-policy actor-critic algorithm that extends DDPG with:
1. Twin Q-networks (clipped double Q-learning)
2. Delayed policy updates
3. Target policy smoothing

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
        self.replay_memory_size = config.get("replay_memory_size", 1_000_000)
        self.warmup_steps = config.get("warmup_steps", 10_000)
        
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

    # --------------------------------------------------
    #  Helper: environment-specific action scaling
    # --------------------------------------------------
    @staticmethod
    def _scale_action_for_env(action: np.ndarray, env, env_name: str | None):
        """
        Convert internal action in [-1,1] to env action.

        For LunarLanderContinuous: Box([-1, -1], [1, 1]) so we keep it.
        For CarRacing: [steer in [-1,1], gas in [0,1], brake in [0,1]].
        """
        if env_name is not None and "CarRacing" in env_name:
            # Internal: a in [-1,1]^3
            a = np.clip(action, -1.0, 1.0).copy()
            env_action = np.zeros_like(a)
            env_action[0] = a[0]                 # steering stays [-1,1]
            env_action[1] = 0.5 * (a[1] + 1.0)   # map [-1,1] -> [0,1]
            env_action[2] = 0.5 * (a[2] + 1.0)   # map [-1,1] -> [0,1]
        else:
            env_action = np.clip(action, -1.0, 1.0)

        # Finally, clip to env bounds (for safety)
        if hasattr(env, "action_space"):
            low = env.action_space.low
            high = env.action_space.high
            env_action = np.clip(env_action, low, high)
        return env_action
    
    # --------------------------------------------------
    #  Action selection
    # --------------------------------------------------
    def select_action(self, state: np.ndarray, deterministic: bool = False, 
                      env_name: str = None) -> np.ndarray:
        """
        Select an *internal* action in [-1,1]^dim using the current policy.
        (Scaling to actual env action is done in train/evaluate.)

        Args:
            state: Current state
            deterministic: If True, return action without exploration noise
            env_name: Unused here, kept for compatibility
        
        Returns:
            action: Selected action in [-1, 1]
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        self.actor.train()

        if not deterministic:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = action + noise

        # Clip to internal range [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        return action
    
    # --------------------------------------------------
    #  Replay buffer sampling
    # --------------------------------------------------
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                         next_state: np.ndarray, done: bool):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def sample_batch(self) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch from the replay buffer.
        
        Returns:
            batch: Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones, dtype=np.float32), dtype=torch.float32, device=self.device).unsqueeze(-1)

        return states, actions, rewards, next_states, dones
    
    # --------------------------------------------------
    #  Target network soft update
    # --------------------------------------------------
    def soft_update(self, target: nn.Module, source: nn.Module):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    # --------------------------------------------------
    #  Networks update step
    # --------------------------------------------------
    def update_networks(self) -> Dict[str, float]:
        """
        Update critic and actor networks using a batch from the replay buffer.
        
        Returns:
            losses: Dictionary containing loss values
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.sample_batch()

        # ------- Critic update (both critics) -------
        with torch.no_grad():
            # Target policy smoothing: μ(s′) + clipped noise
            next_actions = self.actor_target(next_states)
            noise = torch.randn_like(next_actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)

            # Clipped double Q-learning: use min(Q1, Q2)
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_value = rewards + (1.0 - dones) * self.gamma * target_q
            target_value = torch.clamp(target_value, -100.0, 100.0)


        # Critic 1 loss
        current_q1 = self.critic1(states, actions)
        critic1_loss = F.mse_loss(current_q1, target_value)

        # Critic 2 loss
        current_q2 = self.critic2(states, actions)
        critic2_loss = F.mse_loss(current_q2, target_value)

        # Optimize critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        actor_loss_value = None

        # ------- Delayed policy update -------
        if self.total_steps % self.policy_delay == 0:
            # Maximize Q1(s, μ(s)) --> minimize -Q1
            actor_actions = self.actor(states)
            actor_loss = -self.critic1(states, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update all target networks
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic1_target, self.critic1)
            self.soft_update(self.critic2_target, self.critic2)

            actor_loss_value = actor_loss.item()

        losses = {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
        }
        if actor_loss_value is not None:
            losses["actor_loss"] = actor_loss_value

        return losses
    
    # --------------------------------------------------
    #  Training loop (compatible with main.py)
    # --------------------------------------------------
    def train(self, env, config: Dict[str, Any] = None, logger=None) -> Dict[str, Any]:
      """
      Train the TD3 agent on the given environment.

      Args:
          env: Gymnasium environment (wrapped)
          config: Configuration dictionary (optional, uses self.config if None)
          logger: Weights & Biases logger (optional)

      Returns:
          training_stats: Dictionary containing training statistics
      """
      cfg = config or self.config
      episodes = int(cfg.get("episodes", 1000))
      max_episode_steps = int(cfg.get("max_episode_steps", 1000))
      eval_every = int(cfg.get("eval_every", 0))       # 0 = no periodic eval
      eval_episodes = int(cfg.get("eval_episodes", 5))

      episode_rewards = []
      best_mean_reward = -np.inf

      # Get env name for action scaling
      env_name = getattr(env, "env_name", None)
      if env_name is None and hasattr(env, "unwrapped") and hasattr(env.unwrapped, "spec"):
          env_name = env.unwrapped.spec.id

      for episode in range(1, episodes + 1):
          state, _ = env.reset()
          episode_reward = 0.0
          episode_steps = 0

          terminated = False
          truncated = False
          if self.total_steps == 0:
            print("DEBUG OBS:", state, "shape:", state.shape)


          while not (terminated or truncated) and episode_steps < max_episode_steps:
              self.total_steps += 1
              episode_steps += 1

              # -------- 1. ACTION SELECTION -------- #
              if self.total_steps < self.warmup_steps:
                  internal_action = np.random.uniform(-1.0, 1.0, size=self.action_dim)
              else:
                  internal_action = self.select_action(
                      state, deterministic=False, env_name=env_name
                  )

              # Scale for env
              env_action = self._scale_action_for_env(internal_action, env, env_name)

              # -------- 2. STEP ENVIRONMENT -------- #
              next_state, reward, terminated, truncated, info = env.step(env_action)

              # TRUE TERMINAL = terminated
              # TIME LIMIT / TRUNCATION = do NOT cut bootstrapping
              td_done = float(terminated)       # used for TD backup
              episode_done = terminated or truncated   # used for breaking episode

              # -------- 3. STORE TRANSITION -------- #
              self.store_transition(state, internal_action, reward, next_state, td_done)

              state = next_state
              episode_reward += reward

              # -------- 4. UPDATE NETWORKS -------- #
              losses = self.update_networks()

              if logger is not None and losses:
                  logger.log(
                      {
                          "train/critic1_loss": losses.get("critic1_loss"),
                          "train/critic2_loss": losses.get("critic2_loss"),
                          "train/actor_loss": losses.get("actor_loss", 0.0),
                          "train/total_steps": self.total_steps,
                          "train/episode": episode,
                      },
                      step=self.total_steps,
                  )

              if episode_done:
                  break

          # -------- 5. PRINT PROGRESS -------- #
          mean_last_10 = (
              np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
          )
          print(
              f"Episode {episode:4d} | "
              f"Reward: {episode_reward:7.2f} | "
              f"Mean(10): {mean_last_10:7.2f} | "
              f"Steps: {episode_steps}"
          )

          # Record episode reward
          episode_rewards.append(episode_reward)

          if logger is not None:
              logger.log(
                  {
                      "train/episode_reward": episode_reward,
                      "train/episode_length": episode_steps,
                      "train/mean_reward_last_10": mean_last_10,
                  },
                  step=self.total_steps,
              )

          # Optional evaluation
          if eval_every > 0 and episode % eval_every == 0:
              eval_stats = self.evaluate(env, num_episodes=eval_episodes)

              if logger is not None:
                  logger.log(
                      {
                          "eval/mean_reward": eval_stats["mean_reward"],
                          "eval/std_reward": eval_stats["std_reward"],
                      },
                      step=self.total_steps,
                  )

          best_mean_reward = max(best_mean_reward, mean_last_10)

      # -------- 6. FINAL STATISTICS -------- #
      training_stats = {
          "total_episodes": episodes,
          "total_steps": self.total_steps,
          "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
          "best_mean_reward_last_10": float(best_mean_reward),
          "max_episode_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
          "min_episode_reward": float(np.min(episode_rewards)) if episode_rewards else 0.0,
      }
      return training_stats

    
    # --------------------------------------------------
    #  Evaluation
    # --------------------------------------------------
    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the trained policy.
        
        Args:
            env: Gymnasium environment
            num_episodes: Number of episodes to evaluate
            
        Returns:
            eval_stats: Dictionary containing evaluation statistics
        """
        rewards = []

        env_name = getattr(env, "env_name", None)
        if env_name is None and hasattr(env, "unwrapped") and hasattr(env.unwrapped, "spec"):
            env_name = env.unwrapped.spec.id

        for ep in range(1, num_episodes + 1):
            state, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0.0
            steps = 0

            while not (done or truncated):
                internal_action = self.select_action(state, deterministic=True, env_name=env_name)
                env_action = self._scale_action_for_env(internal_action, env, env_name)
                next_state, reward, terminated, truncated, info = env.step(env_action)
                state = next_state
                episode_reward += reward
                done = terminated or truncated
                steps += 1
            print(f"[TEST] Episode {ep}/{num_episodes} | Reward: {episode_reward:.2f}")
            rewards.append(episode_reward)

        rewards = np.array(rewards, dtype=np.float32)
        return {
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "min_reward": float(rewards.min()),
            "max_reward": float(rewards.max()),
            "episode_rewards": rewards.tolist()
        }
    
    # --------------------------------------------------
    #  Save / Load
    # --------------------------------------------------
    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
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
