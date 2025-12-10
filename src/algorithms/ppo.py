"""
Proximal Policy Optimization (PPO) implementation for Assignment 4.

- Compatible with src/train.py, src/test.py, src/record.py, src/environment.py
- Uses configuration from configs/ppo_config.yaml
- Single-environment, continuous action spaces (Box2D: LunarLander-v3, CarRacing-v3)
- Step-based training loop (SB3-style): rollouts of n_steps, multiple epochs
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import os  # >>> NEW

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ============================================================
#  Utility
# ============================================================


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


# ============================================================
#  Actor–Critic Network (MLP, SB3-like)
# ============================================================


class ActorCritic(nn.Module):
    """
    Shared MLP trunk with separate actor (Gaussian) and critic heads.

    This is similar in spirit to SB3's MlpPolicy for continuous actions:
    - Tanh activations in hidden layers
    - Orthogonal initialization
    - Trainable log_std parameter (state-independent diagonal Gaussian)
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(64, 64)):
        super().__init__()

        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.Tanh())
            last_dim = h
        self.shared = nn.Sequential(*layers)

        # Policy head (mean)
        self.mu = nn.Linear(last_dim, act_dim)

        # Log-std as free parameter
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Value head
        self.v = nn.Linear(last_dim, 1)

        # Orthogonal init
        for m in self.shared:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
                nn.init.constant_(m.bias, 0.0)

        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0.0)

        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.constant_(self.v.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        """
        Returns:
            mu: (batch, act_dim)
            std: (batch, act_dim)
            value: (batch,)
        """
        x = self.shared(obs)
        mu = self.mu(x)
        std = self.log_std.exp().expand_as(mu)
        value = self.v(x).squeeze(-1)
        return mu, std, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ):
        """
        Used both for sampling and for computing log_probs/values during training.
        """
        mu, std, value = self.forward(obs)
        dist = Normal(mu, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value


# ============================================================
#  Rollout Buffer with GAE
# ============================================================


@dataclass
class RolloutBuffer:
    n_steps: int
    obs_dim: int
    act_dim: int
    gamma: float
    gae_lambda: float
    device: torch.device

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.obs = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.act_dim), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps,), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps,), dtype=np.float32)
        self.dones = np.zeros((self.n_steps,), dtype=np.float32)
        self.values = np.zeros((self.n_steps,), dtype=np.float32)

        self.advantages = np.zeros((self.n_steps,), dtype=np.float32)
        self.returns = np.zeros((self.n_steps,), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.values[self.pos] = value
        self.pos += 1
        if self.pos >= self.n_steps:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float, last_done: bool):
        """
        GAE(λ) as used in SB3:
        delta_t = r_t + gamma * V_{t+1} * (1 - done_{t+1}) - V_t
        A_t = delta_t + gamma * λ * (1 - done_{t+1}) * A_{t+1}
        """
        last_gae = 0.0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]

            delta = (
                self.rewards[step]
                + self.gamma * next_value * next_non_terminal
                - self.values[step]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae

        self.returns = self.advantages + self.values

    def get(self, batch_size: int):
        """
        Yield mini-batches as torch tensors.
        """
        assert self.full, "RolloutBuffer is not full yet."

        n_steps = self.n_steps
        indices = np.random.permutation(n_steps)

        obs = to_tensor(self.obs, self.device)
        actions = to_tensor(self.actions, self.device)
        log_probs = to_tensor(self.log_probs, self.device)
        advantages = to_tensor(self.advantages, self.device)
        returns = to_tensor(self.returns, self.device)
        values = to_tensor(self.values, self.device)

        # Normalize advantages (SB3 style)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for start in range(0, n_steps, batch_size):
            end = start + batch_size
            mb_idx = indices[start:end]
            yield (
                obs[mb_idx],
                actions[mb_idx],
                log_probs[mb_idx],
                advantages[mb_idx],
                returns[mb_idx],
                values[mb_idx],
            )


# ============================================================
#  PPO Agent (SB3-style, step-based training)
# ============================================================


class PPO:
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Args:
            state_dim: Dimension of observation vector
            action_dim: Dimension of action vector
            config: Dictionary from configs/ppo_config.yaml
        """

        # Device
        device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Hyperparameters (support both original and SB3-style keys)
        self.learning_rate = config.get("learning_rate", 3e-4)

        # discount_factor OR gamma
        self.gamma = config.get("gamma", config.get("discount_factor", 0.99))

        self.gae_lambda = config.get("gae_lambda", 0.95)

        # clip_epsilon OR clip_range
        self.clip_range = config.get("clip_range", config.get("clip_epsilon", 0.2))

        # entropy_coef OR ent_coef (SB3 default is 0.0)
        self.ent_coef = config.get("ent_coef", config.get("entropy_coef", 0.0))

        # value_coef OR vf_coef
        self.vf_coef = config.get("vf_coef", config.get("value_coef", 0.5))

        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.batch_size = config.get("batch_size", 64)
        self.n_steps = config.get("n_steps", 2048)
        self.n_epochs = config.get("n_epochs", 10)

        # Episodes and steps for total_timesteps estimation
        self.episodes = config.get("episodes", 1000)
        self.max_episode_steps = config.get("max_episode_steps", 1000)

        # >>> NEW: early stopping + best-model settings
        self.best_mean_window = config.get("best_mean_window", 50)
        self.early_stop_patience = config.get("early_stop_patience", 80)
        # if best_model_path not provided, derive from save_path or use default
        default_best_path = "models/ppo_best_model.pth"
        self.best_model_path = config.get(
            "best_model_path",
            config.get("save_path", default_best_path).replace(".pth", "_best.pth")
            if config.get("save_path")
            else default_best_path,
        )

        # Actor–critic network
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # Rollout buffer
        self.buffer = RolloutBuffer(
            n_steps=self.n_steps,
            obs_dim=self.state_dim,
            act_dim=self.action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=self.device,
        )

        # Trackers
        self.global_step = 0
        self.total_episodes = 0

    # --------------------------------------------------------
    #  Core interaction methods
    # --------------------------------------------------------

    def _sample_action(self, state: np.ndarray):
        """
        Sample an action from the current policy for a single state.
        Returns (action_np, log_prob_float, value_float).
        """
        state_tensor = to_tensor(state, self.device).unsqueeze(0)  # (1, obs_dim)
        with torch.no_grad():
            action, log_prob, _, value = self.policy.get_action_and_value(state_tensor)
        action = action.squeeze(0).cpu().numpy()
        log_prob = float(log_prob.item())
        value = float(value.item())
        return action, log_prob, value

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """
        Used by record.py and test scripts:
        - deterministic=True => use mean of Gaussian
        - deterministic=False => sample from policy
        """
        state_tensor = to_tensor(state, self.device).unsqueeze(0)
        with torch.no_grad():
            mu, std, value = self.policy.forward(state_tensor)
            if deterministic:
                action = mu
            else:
                dist = Normal(mu, std)
                action = dist.sample()

        action = action.squeeze(0).cpu().numpy()
        return action

    # --------------------------------------------------------
    #  Rollout collection and PPO update
    # --------------------------------------------------------

    def _collect_rollout(self, env) -> List[float]:
        """
        Collect n_steps transitions into the rollout buffer.
        Returns list of completed episode rewards during this rollout.
        """
        self.buffer.reset()
        episode_rewards = []
        episode_reward = 0.0

        # Start from a fresh state *for each rollout*
        state, _ = env.reset()
        done = False
        truncated = False

        for step in range(self.n_steps):
            self.global_step += 1

            action, log_prob, value = self._sample_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            terminal = bool(done or truncated)

            self.buffer.add(
                obs=state,
                action=action,
                log_prob=log_prob,
                reward=reward,
                done=terminal,
                value=value,
            )

            state = next_state
            episode_reward += reward

            if terminal:
                self.total_episodes += 1
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                state, _ = env.reset()
                done = False
                truncated = False

        # Bootstrap value from last state for GAE
        with torch.no_grad():
            state_tensor = to_tensor(state, self.device).unsqueeze(0)
            _, _, _, last_value = self.policy.get_action_and_value(state_tensor)
        last_value = float(last_value.item())

        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            last_done=bool(done or truncated),
        )

        return episode_rewards

    def _update(self):
        """
        PPO update over the rollout buffer for n_epochs.
        """
        for _ in range(self.n_epochs):
            for (
                obs_batch,
                actions_batch,
                old_log_probs_batch,
                advantages_batch,
                returns_batch,
                _,
            ) in self.buffer.get(self.batch_size):

                # New log_probs, values, entropy
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    obs_batch, actions_batch
                )

                log_ratio = new_log_probs - old_log_probs_batch
                ratio = torch.exp(log_ratio)

                # Policy loss (clipped surrogate)
                unclipped = ratio * advantages_batch
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                ) * advantages_batch
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value loss (no value clipping here, but could be added to be even closer to SB3)
                value_loss = F.mse_loss(new_values, returns_batch)

                # Entropy bonus
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - self.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    # --------------------------------------------------------
    #  Helper: save best model
    # --------------------------------------------------------  # >>> NEW

    def _save_best_model(self):
        if not self.best_model_path:
            return
        directory = os.path.dirname(self.best_model_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.policy.state_dict(), self.best_model_path)

    # --------------------------------------------------------
    #  Public API: train / evaluate / save / load
    # --------------------------------------------------------

    def train(self, env, config: Dict[str, Any], logger=None) -> Dict[str, Any]:
        """
        Main training loop (called from src/train.py).

        We convert (episodes, max_episode_steps) → total_timesteps,
        then compute number of PPO updates: total_timesteps // n_steps.
        """
        episodes = config.get("episodes", self.episodes)
        max_steps = config.get("max_episode_steps", self.max_episode_steps)

        total_timesteps = episodes * max_steps
        num_updates = max(1, total_timesteps // self.n_steps)

        all_episode_rewards: List[float] = []

        print(
            f"PPO Training: total_timesteps ≈ {total_timesteps} "
            f"({num_updates} updates × {self.n_steps} steps)"
        )

        best_mean_reward_10 = -np.inf
        best_mean_reward_window = -np.inf  # >>> NEW
        updates_without_improvement = 0    # >>> NEW
        window = self.best_mean_window     # >>> NEW

        for update in range(1, num_updates + 1):
            ep_rewards = self._collect_rollout(env)
            all_episode_rewards.extend(ep_rewards)

            self._update()

            # Logging / printing
            if len(all_episode_rewards) > 0:
                last_10 = all_episode_rewards[-10:]
                mean_10 = float(np.mean(last_10))

                # >>> NEW: mean over window (default 50)
                if len(all_episode_rewards) >= window:
                    last_w = all_episode_rewards[-window:]
                    mean_w = float(np.mean(last_w))
                else:
                    mean_w = float(np.mean(all_episode_rewards))
            else:
                mean_10 = 0.0
                mean_w = 0.0

            # Track best mean over 10 (for backwards compatibility)
            if mean_10 > best_mean_reward_10:
                best_mean_reward_10 = mean_10

            # >>> NEW: best mean over window (50)
            if mean_w > best_mean_reward_window:
                best_mean_reward_window = mean_w
                updates_without_improvement = 0
                self._save_best_model()
            else:
                updates_without_improvement += 1

            print(
                f"[Update {update}/{num_updates}] "
                f"Global steps: {self.global_step}, "
                f"Episodes: {self.total_episodes}, "
                f"Mean reward (last 10): {mean_10:.2f}, "
                f"Mean reward (last {window}): {mean_w:.2f}, "
                f"Best mean (last {window}): {best_mean_reward_window:.2f}"
            )

            if logger is not None:
                log_dict = {
                    "train/update": update,
                    "train/global_steps": self.global_step,
                    "train/episodes": self.total_episodes,
                    "train/mean_reward_10": mean_10,
                    "train/best_mean_reward_10": best_mean_reward_10,
                    "train/mean_reward_window": mean_w,
                    f"train/best_mean_reward_{window}": best_mean_reward_window,
                }
                logger.log(log_dict)

            # >>> NEW: early stopping
            if (
                self.early_stop_patience is not None
                and updates_without_improvement >= self.early_stop_patience
            ):
                print(
                    f"\nEarly stopping triggered at update {update}. "
                    f"Best mean reward (last {window}): {best_mean_reward_window:.2f}"
                )
                break

        # Final statistics for train.py
        if len(all_episode_rewards) > 0:
            rewards_array = np.array(all_episode_rewards, dtype=np.float32)
            stats = {
                "total_episodes": int(self.total_episodes),
                "total_steps": int(self.global_step),
                "mean_reward": float(rewards_array.mean()),
                "std_reward": float(rewards_array.std()),
                "min_reward": float(rewards_array.min()),
                "max_reward": float(rewards_array.max()),
                "best_mean_reward_10": float(best_mean_reward_10),
                f"best_mean_reward_{window}": float(best_mean_reward_window),
            }
        else:
            stats = {
                "total_episodes": int(self.total_episodes),
                "total_steps": int(self.global_step),
            }

        return stats

    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluation loop used by src/test.py.
        """
        episode_rewards: List[float] = []
        episode_durations: List[int] = []

        for ep in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            ep_ret = 0.0
            steps = 0

            while not (done or truncated):
                action = self.select_action(state, deterministic=True)
                state, reward, done, truncated, _ = env.step(action)
                ep_ret += reward
                steps += 1

            episode_rewards.append(ep_ret)
            episode_durations.append(steps)
            print(
                f"[Eval] Episode {ep + 1}/{num_episodes} | "
                f"Return: {ep_ret:.2f} | Steps: {steps}"
            )

        rewards_array = np.array(episode_rewards, dtype=np.float32)
        durations_array = np.array(episode_durations, dtype=np.int32)

        stats = {
            "episode_rewards": episode_rewards,
            "episode_durations": episode_durations,
            "mean_reward": float(rewards_array.mean()),
            "std_reward": float(rewards_array.std()),
            "min_reward": float(rewards_array.min()),
            "max_reward": float(rewards_array.max()),
            "mean_duration": float(durations_array.mean()),
        }
        return stats

    # --------------------------------------------------------
    #  Saving / Loading (used by train.py & test.py)
    # --------------------------------------------------------

    def save(self, path: str):
        """
        Save model parameters.
        """
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        """
        Load model parameters.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
