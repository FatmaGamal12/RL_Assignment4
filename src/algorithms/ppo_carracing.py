"""
PPO for CarRacing-v3 (continuous) – SOTA-style

- Observation: raw 96x96x3 images from Gymnasium
- Preprocessing inside the agent:
    * RGB → grayscale
    * Normalized to [0, 1]
    * Frame stacking (num_stack frames)
    * Optional frame skipping
- Policy: NatureCNN + Gaussian actor-critic
- GAE + clipped PPO objective
- Reward window (mean over last 50 episodes)
- Checkpoint: save best model when mean_reward_window improves
- Early stopping based on reward_window plateau
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

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


def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    """
    Convert RGB (96x96x3) → grayscale (96x96), normalized to [0, 1].

    Args:
        obs: np.ndarray with shape (H, W, 3), dtype uint8 or float32

    Returns:
        gray: np.ndarray with shape (96, 96), float32 in [0, 1]
    """
    # Ensure float32
    obs = obs.astype(np.float32)

    # Luma transform (ITU-R BT.601)
    gray = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]

    # Normalize to [0, 1]
    gray /= 255.0
    return gray.astype(np.float32)


# ============================================================
#  NatureCNN Actor–Critic for stacked frames
# ============================================================

class NatureCNNActorCritic(nn.Module):
    """
    NatureCNN-style shared trunk + separate actor/critic heads.

    Input: (B, C, 96, 96)
        C = num_stack (e.g., 4)
    """

    def __init__(self, num_stack: int, action_dim: int):
        super().__init__()
        self.num_stack = num_stack
        self.action_dim = action_dim

        # Nature CNN conv stack
        self.conv = nn.Sequential(
            nn.Conv2d(num_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Compute flatten size with dummy input (96x96)
        with torch.no_grad():
            dummy = torch.zeros(1, num_stack, 96, 96)
            conv_out = self.conv(dummy)
            self.flatten_size = conv_out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
        )

        # Actor head (Gaussian policy)
        self.mu = nn.Linear(512, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.v = nn.Linear(512, 1)

        # Orthogonal initialization (SB3-style)
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
                nn.init.constant_(m.bias, 0.0)

        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
                nn.init.constant_(m.bias, 0.0)

        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0.0)

        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.constant_(self.v.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: (B, C, H, W) float32

        Returns:
            mu: (B, action_dim)
            std: (B, action_dim)
            value: (B,)
        """
        x = self.conv(obs)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

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
        Sample or evaluate action + log_prob + entropy + value.
        """
        mu, std, value = self.forward(obs)
        dist = Normal(mu, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value


# ============================================================
#  Rollout Buffer for image observations
# ============================================================

@dataclass
class ImageRolloutBuffer:
    n_steps: int
    obs_shape: Tuple[int, int, int]  # (C, H, W)
    act_dim: int
    gamma: float
    gae_lambda: float
    device: torch.device

    def __post_init__(self):
        self.reset()

    def reset(self):
        C, H, W = self.obs_shape
        self.obs = np.zeros((self.n_steps, C, H, W), dtype=np.float32)
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
        Standard GAE(λ).
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
        Yield minibatches as torch tensors.
        """
        assert self.full, "ImageRolloutBuffer is not full yet."

        indices = np.random.permutation(self.n_steps)

        obs = to_tensor(self.obs, self.device)
        actions = to_tensor(self.actions, self.device)
        log_probs = to_tensor(self.log_probs, self.device)
        advantages = to_tensor(self.advantages, self.device)
        returns = to_tensor(self.returns, self.device)
        values = to_tensor(self.values, self.device)

        # Normalize advantages (SB3-style)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for start in range(0, self.n_steps, batch_size):
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
#  PPO for CarRacing-v3 with stacked frames
# ============================================================

class PPO_CarRacing:
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        We ignore state_dim (image env) and rely on known CarRacing obs shape.

        Args:
            state_dim: Unused – kept for API compatibility.
            action_dim: Dimension of continuous action vector (3 for CarRacing).
            config: Config dict from ppo_config.yaml (CarRacing-v3 section).
        """

        # Device
        device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)

        # Hyperparameters
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", config.get("discount_factor", 0.99))
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_range = config.get("clip_range", config.get("clip_epsilon", 0.2))
        self.ent_coef = config.get("ent_coef", config.get("entropy_coef", 0.01))
        self.vf_coef = config.get("vf_coef", config.get("value_coef", 0.5))
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.batch_size = config.get("batch_size", 64)
        self.n_steps = config.get("n_steps", 2048)
        self.n_epochs = config.get("n_epochs", 10)

        self.episodes = config.get("episodes", 1500)
        self.max_episode_steps = config.get("max_episode_steps", 1000)

        # Frame-related hyperparams
        self.num_stack = config.get("frame_stack", 4)
        self.frame_skip = config.get("frame_skip", 2)

        # Early stopping & checkpoint
        self.reward_window = config.get("reward_window", 50)
        self.early_stop_patience = config.get("early_stop_patience", 100)
        self.early_stop_min_episodes = config.get("early_stop_min_episodes", 400)
        self.checkpoint_path = config.get(
            "checkpoint_path", "models/ppo_CarRacing-v3_best.pth"
        )

        # Assumed obs size after preprocessing: (num_stack, 96, 96)
        self.obs_shape = (self.num_stack, 96, 96)
        self.action_dim = action_dim

        # Policy network
        self.policy = NatureCNNActorCritic(
            num_stack=self.num_stack,
            action_dim=self.action_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # Rollout buffer
        self.buffer = ImageRolloutBuffer(
            n_steps=self.n_steps,
            obs_shape=self.obs_shape,
            act_dim=self.action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=self.device,
        )

        # Trackers
        self.global_step = 0
        self.total_episodes = 0

    # --------------------------------------------------------
    #  Frame stacking helpers
    # --------------------------------------------------------

    def _init_stack(self, first_obs: np.ndarray) -> np.ndarray:
        """Create initial stack by repeating the first preprocessed frame."""
        frame = preprocess_observation(first_obs)  # (H, W)
        stack = np.repeat(frame[None, :, :], self.num_stack, axis=0)  # (C, H, W)
        return stack.astype(np.float32)

    def _update_stack(self, stack: np.ndarray, new_obs: np.ndarray) -> np.ndarray:
        """Append new frame to stack and drop oldest."""
        frame = preprocess_observation(new_obs)  # (H, W)
        frame = frame[None, :, :]  # (1, H, W)
        new_stack = np.concatenate([stack[1:], frame], axis=0)
        return new_stack.astype(np.float32)

    # --------------------------------------------------------
    #  Core interaction
    # --------------------------------------------------------

    def _sample_action(self, state_stack: np.ndarray):
        """
        state_stack: (C, H, W) numpy array
        """
        obs_tensor = to_tensor(state_stack, self.device).unsqueeze(0)  # (1, C, H, W)
        with torch.no_grad():
            action, log_prob, _, value = self.policy.get_action_and_value(obs_tensor)
        action_np = action.squeeze(0).cpu().numpy()
        return action_np, float(log_prob.item()), float(value.item())

    def select_action(self, state_stack: np.ndarray, deterministic: bool = False):
        """
        For evaluation/recording with a trained model.

        state_stack: (C, H, W)
        """
        obs_tensor = to_tensor(state_stack, self.device).unsqueeze(0)
        with torch.no_grad():
            mu, std, _ = self.policy.forward(obs_tensor)
            if deterministic:
                action = mu
            else:
                dist = Normal(mu, std)
                action = dist.sample()
        return action.squeeze(0).cpu().numpy()

    # --------------------------------------------------------
    #  Rollout collection with frame skip & reward shaping
    # --------------------------------------------------------

    def _car_racing_reward_shaping(self, reward: float, info: Dict[str, Any]) -> float:
        """
        Simple reward shaping hook (you can extend this later).
        For now, just passthrough.

        You can use info dict if you want (e.g., grass penalty).
        """
        return reward

    def _collect_rollout(self, env) -> List[float]:
        self.buffer.reset()
        episode_rewards: List[float] = []
        episode_reward = 0.0

        # Reset env and init stack
        obs, _ = env.reset()
        state_stack = self._init_stack(obs)

        done = False
        truncated = False

        for step in range(self.n_steps):
            self.global_step += 1

            # Sample action from stacked frames
            action, log_prob, value = self._sample_action(state_stack)

            # Frame skip: repeat same action self.frame_skip times
            total_reward = 0.0
            for _ in range(self.frame_skip):
                next_obs, reward, done, truncated, info = env.step(action)
                reward = self._car_racing_reward_shaping(reward, info)
                total_reward += reward
                if done or truncated:
                    break

            terminal = bool(done or truncated)

            # Store transition with aggregated reward
            self.buffer.add(
                obs=state_stack,
                action=action,
                log_prob=log_prob,
                reward=total_reward,
                done=terminal,
                value=value,
            )

            episode_reward += total_reward

            # Prepare next state stack
            if terminal:
                self.total_episodes += 1
                episode_rewards.append(episode_reward)
                episode_reward = 0.0

                obs, _ = env.reset()
                state_stack = self._init_stack(obs)
                done = False
                truncated = False
            else:
                state_stack = self._update_stack(state_stack, next_obs)

        # Bootstrap value from last stack
        with torch.no_grad():
            obs_tensor = to_tensor(state_stack, self.device).unsqueeze(0)
            _, _, _, last_value = self.policy.get_action_and_value(obs_tensor)
        last_value = float(last_value.item())

        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            last_done=bool(done or truncated),
        )

        return episode_rewards

    # --------------------------------------------------------
    #  PPO update
    # --------------------------------------------------------

    def _update(self):
        for _ in range(self.n_epochs):
            for (
                obs_batch,
                actions_batch,
                old_log_probs_batch,
                advantages_batch,
                returns_batch,
                _,
            ) in self.buffer.get(self.batch_size):

                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    obs_batch, actions_batch
                )

                log_ratio = new_log_probs - old_log_probs_batch
                ratio = torch.exp(log_ratio)

                # Policy loss
                unclipped = ratio * advantages_batch
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                ) * advantages_batch
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value loss
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
    #  Public API: train / evaluate / save / load
    # --------------------------------------------------------

    def train(self, env, config: Dict[str, Any], logger=None) -> Dict[str, Any]:
        episodes = config.get("episodes", self.episodes)
        max_steps = config.get("max_episode_steps", self.max_episode_steps)

        total_timesteps = episodes * max_steps
        num_updates = max(1, total_timesteps // self.n_steps)

        all_episode_rewards: List[float] = []

        print(
            f"PPO CarRacing Training: total_timesteps ≈ {total_timesteps} "
            f"({num_updates} updates × {self.n_steps} steps, frame_skip={self.frame_skip})"
        )

        best_mean_reward_10 = -np.inf
        best_mean_reward_window = -np.inf
        no_improve_updates = 0

        for update in range(1, num_updates + 1):
            ep_rewards = self._collect_rollout(env)
            all_episode_rewards.extend(ep_rewards)

            self._update()

            if len(all_episode_rewards) > 0:
                last_10 = all_episode_rewards[-10:]
                mean_10 = float(np.mean(last_10))

                window_size = min(self.reward_window, len(all_episode_rewards))
                window_rewards = all_episode_rewards[-window_size:]
                mean_window = float(np.mean(window_rewards))
            else:
                mean_10 = 0.0
                mean_window = 0.0

            # Track best metrics
            if mean_10 > best_mean_reward_10:
                best_mean_reward_10 = mean_10

            improved = False
            if mean_window > best_mean_reward_window:
                best_mean_reward_window = mean_window
                improved = True
                no_improve_updates = 0
                # Save checkpoint whenever we get a new best over the window
                self.save(self.checkpoint_path)
            else:
                no_improve_updates += 1

            print(
                f"[Update {update}/{num_updates}] "
                f"Global steps: {self.global_step}, "
                f"Episodes: {self.total_episodes}, "
                f"Mean reward (last 10): {mean_10:.2f}, "
                f"Mean reward (last {self.reward_window}): {mean_window:.2f}, "
                f"Best mean_10: {best_mean_reward_10:.2f}, "
                f"Best mean_{self.reward_window}: {best_mean_reward_window:.2f}, "
                f"No-improve updates: {no_improve_updates}"
            )

            if logger is not None:
                logger.log(
                    {
                        "train/update": update,
                        "train/global_steps": self.global_step,
                        "train/episodes": self.total_episodes,
                        "train/mean_reward_10": mean_10,
                        "train/mean_reward_window": mean_window,
                        "train/best_mean_reward_10": best_mean_reward_10,
                        "train/best_mean_reward_50": best_mean_reward_window,
                    }
                )

            # Early stopping if plateaued for too long
            if (
                self.total_episodes >= self.early_stop_min_episodes
                and no_improve_updates >= self.early_stop_patience
            ):
                print(
                    f"\nEarly stopping triggered at update {update}. "
                    f"Best mean reward (last {self.reward_window}): "
                    f"{best_mean_reward_window:.2f}"
                )
                break

        # Final stats
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
                "best_mean_reward_50": float(best_mean_reward_window),
            }
        else:
            stats = {
                "total_episodes": int(self.total_episodes),
                "total_steps": int(self.global_step),
            }

        return stats

    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Standard evaluation loop. Handles frame stacking internally.
        """
        episode_rewards: List[float] = []
        episode_durations: List[int] = []

        for ep in range(num_episodes):
            obs, _ = env.reset()
            state_stack = self._init_stack(obs)

            done = False
            truncated = False
            ep_ret = 0.0
            steps = 0

            while not (done or truncated):
                action = self.select_action(state_stack, deterministic=True)

                total_reward = 0.0
                for _ in range(self.frame_skip):
                    next_obs, reward, done, truncated, info = env.step(action)
                    reward = self._car_racing_reward_shaping(reward, info)
                    total_reward += reward
                    if done or truncated:
                        break

                ep_ret += total_reward
                steps += 1

                if done or truncated:
                    break

                state_stack = self._update_stack(state_stack, next_obs)

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
    #  Saving / Loading
    # --------------------------------------------------------

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
