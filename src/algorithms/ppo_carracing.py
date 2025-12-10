"""
PPO for CarRacing-v3 (continuous) – FIXED SOTA Implementation

Key fixes:
- Reward normalization (VecNormalize)
- State-dependent exploration (SDE)
- Parallel environments (VecEnv)
- Correct hyperparameters from RL Zoo
- 64x64 grayscale preprocessing
- Frame stack: 2 (not 4)
- GELU activation
- Linear learning rate decay
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import math

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


def linear_schedule(initial_value: float):
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# ============================================================
#  Running Mean/Std for Reward Normalization
# ============================================================

class RunningMeanStd:
    """Tracks running mean and std of a stream of data."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


# ============================================================
#  NatureCNN Actor–Critic with GELU and SDE
# ============================================================

class NatureCNNActorCritic(nn.Module):
    """
    NatureCNN for 64x64 grayscale stacked frames with:
    - GELU activation
    - NO orthogonal initialization (ortho_init=False)
    - State-dependent exploration (SDE)
    - Separate 256-dim heads for actor/critic
    """

    def __init__(self, num_stack: int, action_dim: int, use_sde: bool = True, log_std_init: float = -2.0):
        super().__init__()
        self.num_stack = num_stack
        self.action_dim = action_dim
        self.use_sde = use_sde

        # Nature CNN for 64x64 input
        self.conv = nn.Sequential(
            nn.Conv2d(num_stack, 32, kernel_size=8, stride=4),  # -> (32, 15, 15)
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),         # -> (64, 6, 6)
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),         # -> (64, 4, 4)
            nn.GELU(),
        )

        # Compute flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, num_stack, 64, 64)
            conv_out = self.conv(dummy)
            self.flatten_size = conv_out.view(1, -1).shape[1]

        # Shared features
        self.features = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.GELU(),
        )

        # Actor head (256-dim)
        self.actor_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
        )
        self.mu = nn.Linear(256, action_dim)
        
        if use_sde:
            # State-dependent exploration
            self.log_std = nn.Linear(256, action_dim)
        else:
            self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

        # Critic head (256-dim)
        self.critic_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
        )
        self.v = nn.Linear(256, 1)

        # NO orthogonal init (as per RL Zoo tuned params)
        self._init_weights(log_std_init)

    def _init_weights(self, log_std_init):
        # Standard initialization (NOT orthogonal)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # Small init for policy output
        nn.init.uniform_(self.mu.weight, -0.01, 0.01)
        nn.init.constant_(self.mu.bias, 0.0)
        
        if self.use_sde:
            nn.init.constant_(self.log_std.weight, 0.0)
            nn.init.constant_(self.log_std.bias, log_std_init)

    def forward(self, obs: torch.Tensor):
        x = self.conv(obs)
        x = x.view(x.size(0), -1)
        features = self.features(x)

        # Actor
        actor_features = self.actor_head(features)
        mu = self.mu(actor_features)
        
        if self.use_sde:
            log_std = self.log_std(actor_features)
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()
        else:
            std = self.log_std.exp().expand_as(mu)

        # Critic
        critic_features = self.critic_head(features)
        value = self.v(critic_features).squeeze(-1)

        return mu, std, value

    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None):
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
    n_envs: int
    obs_shape: Tuple[int, int, int]
    act_dim: int
    gamma: float
    gae_lambda: float
    device: torch.device

    def __post_init__(self):
        self.reset()

    def reset(self):
        C, H, W = self.obs_shape
        self.obs = np.zeros((self.n_steps, self.n_envs, C, H, W), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_envs, self.act_dim), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)

        self.advantages = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.pos += 1
        if self.pos >= self.n_steps:
            self.full = True

    def compute_returns_and_advantages(self, last_values: np.ndarray, last_dones: np.ndarray):
        """GAE(λ) for vectorized environments."""
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae

        self.returns = self.advantages + self.values

    def get(self, batch_size: int):
        """Yield minibatches."""
        assert self.full, "Buffer not full"
        
        # Flatten batch dimension
        b_obs = self.obs.reshape((-1,) + self.obs.shape[2:])
        b_actions = self.actions.reshape((-1, self.act_dim))
        b_log_probs = self.log_probs.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)

        indices = np.random.permutation(self.n_steps * self.n_envs)

        # Convert to tensors
        obs_t = to_tensor(b_obs, self.device)
        actions_t = to_tensor(b_actions, self.device)
        log_probs_t = to_tensor(b_log_probs, self.device)
        advantages_t = to_tensor(b_advantages, self.device)
        returns_t = to_tensor(b_returns, self.device)
        values_t = to_tensor(b_values, self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            mb_idx = indices[start:end]
            yield (
                obs_t[mb_idx],
                actions_t[mb_idx],
                log_probs_t[mb_idx],
                advantages_t[mb_idx],
                returns_t[mb_idx],
                values_t[mb_idx],
            )


# ============================================================
#  PPO for CarRacing-v3 - FIXED VERSION
# ============================================================

class PPO_CarRacing:
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Fixed PPO implementation with:
        - Reward normalization
        - SDE
        - Parallel environments support
        - Linear LR schedule
        """
        # Device
        device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)

        # Hyperparameters
        self.learning_rate_initial = config.get("learning_rate", 1e-4)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_range = config.get("clip_range", 0.2)
        self.ent_coef = config.get("ent_coef", 0.0)
        self.vf_coef = config.get("vf_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.batch_size = config.get("batch_size", 128)
        self.n_steps = config.get("n_steps", 512)
        self.n_epochs = config.get("n_epochs", 10)
        self.n_envs = config.get("n_envs", 8)

        self.episodes = config.get("episodes", 2000)
        self.max_episode_steps = config.get("max_episode_steps", 1000)

        # Frame-related
        self.num_stack = config.get("frame_stack", 2)
        self.frame_skip = config.get("frame_skip", 2)
        self.image_size = config.get("image_size", 64)

        # SDE
        self.use_sde = config.get("use_sde", True)
        self.sde_sample_freq = config.get("sde_sample_freq", 4)
        self.log_std_init = config.get("log_std_init", -2.0)

        # Reward normalization
        self.normalize_reward = config.get("normalize_reward", True)
        if self.normalize_reward:
            self.reward_rms = RunningMeanStd(shape=())
            self.reward_scale = config.get("reward_scale", 1.0)
            self.clip_reward = config.get("clip_reward", 10.0)

        # Linear LR schedule
        self.use_linear_lr_decay = config.get("use_linear_lr_decay", True)

        # Early stopping & checkpoint
        self.reward_window = config.get("reward_window", 100)
        self.early_stop_patience = config.get("early_stop_patience", 200)
        self.early_stop_min_episodes = config.get("early_stop_min_episodes", 400)
        self.checkpoint_path = config.get("checkpoint_path", "models/ppo_CarRacing-v3_best.pth")

        # Obs shape: (num_stack, 64, 64)
        self.obs_shape = (self.num_stack, self.image_size, self.image_size)
        self.action_dim = action_dim

        # Policy network
        self.policy = NatureCNNActorCritic(
            num_stack=self.num_stack,
            action_dim=self.action_dim,
            use_sde=self.use_sde,
            log_std_init=self.log_std_init,
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate_initial)

        # LR scheduler
        if self.use_linear_lr_decay:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, 
                lr_lambda=lambda epoch: 1.0  # Will be updated manually
            )

        # Rollout buffer
        self.buffer = ImageRolloutBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            obs_shape=self.obs_shape,
            act_dim=self.action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=self.device,
        )

        # Trackers
        self.global_step = 0
        self.total_episodes = 0
        self.num_updates = 0

    def _normalize_reward(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize rewards using running statistics."""
        if not self.normalize_reward:
            return rewards
        
        self.reward_rms.update(rewards)
        normalized = rewards / np.sqrt(self.reward_rms.var + 1e-8)
        normalized = np.clip(normalized, -self.clip_reward, self.clip_reward)
        return normalized * self.reward_scale

    def _update_learning_rate(self, progress_remaining: float):
        """Update learning rate based on progress."""
        if self.use_linear_lr_decay:
            new_lr = self.learning_rate_initial * progress_remaining
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert RGB to grayscale and resize to 64x64."""
        from PIL import Image
        
        # Convert to grayscale
        gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        
        # Resize to 64x64
        img = Image.fromarray(gray.astype(np.uint8))
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Normalize to [0, 1]
        processed = np.array(img, dtype=np.float32) / 255.0
        return processed

    def _init_stack(self, first_obs: np.ndarray) -> np.ndarray:
        """Create initial stack."""
        frame = self._preprocess_frame(first_obs)
        stack = np.repeat(frame[None, :, :], self.num_stack, axis=0)
        return stack.astype(np.float32)

    def _update_stack(self, stack: np.ndarray, new_obs: np.ndarray) -> np.ndarray:
        """Update stack with new frame."""
        frame = self._preprocess_frame(new_obs)
        new_stack = np.concatenate([stack[1:], frame[None, :, :]], axis=0)
        return new_stack.astype(np.float32)

    def _sample_actions(self, state_stacks: np.ndarray):
        """Sample actions for vectorized envs."""
        obs_tensor = to_tensor(state_stacks, self.device)
        with torch.no_grad():
            actions, log_probs, _, values = self.policy.get_action_and_value(obs_tensor)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy()
        )

    def select_action(self, state_stack: np.ndarray, deterministic: bool = False):
        """For evaluation."""
        obs_tensor = to_tensor(state_stack, self.device).unsqueeze(0)
        with torch.no_grad():
            mu, std, _ = self.policy.forward(obs_tensor)
            if deterministic:
                action = mu
            else:
                dist = Normal(mu, std)
                action = dist.sample()
        return action.squeeze(0).cpu().numpy()

    def _collect_rollout(self, envs) -> List[float]:
        """Collect rollout from vectorized environments."""
        self.buffer.reset()
        episode_rewards = []
        episode_counts = np.zeros(self.n_envs)
        episode_returns = np.zeros(self.n_envs)

        # Reset all envs and init stacks
        obs_list = []
        for i in range(self.n_envs):
            obs, _ = envs[i].reset()
            obs_list.append(self._init_stack(obs))
        state_stacks = np.array(obs_list)

        for step in range(self.n_steps):
            self.global_step += self.n_envs

            # Sample actions
            actions, log_probs, values = self._sample_actions(state_stacks)

            # Step all environments
            next_obs_list = []
            raw_rewards = []
            dones = []
            
            for i in range(self.n_envs):
                total_reward = 0.0
                for _ in range(self.frame_skip):
                    next_obs, reward, done, truncated, info = envs[i].step(actions[i])
                    total_reward += reward
                    if done or truncated:
                        break
                
                raw_rewards.append(total_reward)
                dones.append(done or truncated)
                
                episode_counts[i] += 1
                episode_returns[i] += total_reward
                
                if done or truncated:
                    episode_rewards.append(episode_returns[i])
                    episode_returns[i] = 0.0
                    episode_counts[i] = 0
                    self.total_episodes += 1
                    next_obs, _ = envs[i].reset()
                    next_obs_list.append(self._init_stack(next_obs))
                else:
                    next_obs_list.append(self._update_stack(state_stacks[i], next_obs))

            # Normalize rewards
            normalized_rewards = self._normalize_reward(np.array(raw_rewards))

            # Store transition
            self.buffer.add(
                obs=state_stacks,
                action=actions,
                log_prob=log_probs,
                reward=normalized_rewards,
                done=np.array(dones, dtype=np.float32),
                value=values,
            )

            state_stacks = np.array(next_obs_list)

        # Bootstrap values
        with torch.no_grad():
            obs_tensor = to_tensor(state_stacks, self.device)
            _, _, last_values = self.policy.forward(obs_tensor)
        last_values = last_values.cpu().numpy()

        self.buffer.compute_returns_and_advantages(
            last_values=last_values,
            last_dones=dones,
        )

        return episode_rewards

    def _update(self):
        """PPO update."""
        for epoch in range(self.n_epochs):
            for (obs_batch, actions_batch, old_log_probs_batch,
                 advantages_batch, returns_batch, _) in self.buffer.get(self.batch_size):

                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    obs_batch, actions_batch
                )

                log_ratio = new_log_probs - old_log_probs_batch
                ratio = torch.exp(log_ratio)

                # Policy loss
                unclipped = ratio * advantages_batch
                clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages_batch
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, returns_batch)

                # Entropy bonus
                entropy_loss = entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def train(self, envs, config: Dict[str, Any], logger=None) -> Dict[str, Any]:
        """Training loop with vectorized environments."""
        episodes = config.get("episodes", self.episodes)
        max_steps = config.get("max_episode_steps", self.max_episode_steps)

        total_timesteps = episodes * max_steps
        num_updates = max(1, total_timesteps // (self.n_steps * self.n_envs))

        all_episode_rewards = []

        print(f"PPO CarRacing Training (FIXED):")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"  Num updates: {num_updates}")
        print(f"  n_envs: {self.n_envs}, n_steps: {self.n_steps}")
        print(f"  Reward normalization: {self.normalize_reward}")
        print(f"  SDE: {self.use_sde}")

        best_mean_reward = -np.inf
        no_improve_updates = 0

        for update in range(1, num_updates + 1):
            self.num_updates = update
            
            # Update learning rate
            progress_remaining = 1.0 - (update / num_updates)
            self._update_learning_rate(progress_remaining)

            # Collect rollout
            ep_rewards = self._collect_rollout(envs)
            all_episode_rewards.extend(ep_rewards)

            # Update policy
            self._update()

            # Stats
            if len(all_episode_rewards) > 0:
                window_size = min(self.reward_window, len(all_episode_rewards))
                mean_reward = float(np.mean(all_episode_rewards[-window_size:]))
            else:
                mean_reward = 0.0

            improved = False
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                improved = True
                no_improve_updates = 0
                self.save(self.checkpoint_path)
            else:
                no_improve_updates += 1

            if update % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"[Update {update}/{num_updates}] "
                      f"Episodes: {self.total_episodes}, "
                      f"Mean reward (last {window_size}): {mean_reward:.2f}, "
                      f"Best: {best_mean_reward:.2f}, "
                      f"LR: {current_lr:.2e}")

            if logger:
                logger.log({
                    "train/update": update,
                    "train/episodes": self.total_episodes,
                    "train/mean_reward": mean_reward,
                    "train/best_mean_reward": best_mean_reward,
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                })

            # Early stopping
            if (self.total_episodes >= self.early_stop_min_episodes and
                no_improve_updates >= self.early_stop_patience):
                print(f"\nEarly stopping at update {update}")
                break

        # Final stats
        if len(all_episode_rewards) > 0:
            rewards_array = np.array(all_episode_rewards)
            stats = {
                "total_episodes": self.total_episodes,
                "mean_reward": float(rewards_array.mean()),
                "std_reward": float(rewards_array.std()),
                "best_mean_reward": float(best_mean_reward),
            }
        else:
            stats = {"total_episodes": self.total_episodes}

        return stats

    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluation loop."""
        episode_rewards = []

        for ep in range(num_episodes):
            obs, _ = env.reset()
            state_stack = self._init_stack(obs)
            done = False
            ep_ret = 0.0

            while not done:
                action = self.select_action(state_stack, deterministic=True)
                
                total_reward = 0.0
                for _ in range(self.frame_skip):
                    next_obs, reward, done, truncated, info = env.step(action)
                    total_reward += reward
                    if done or truncated:
                        done = True
                        break

                ep_ret += total_reward
                if not done:
                    state_stack = self._update_stack(state_stack, next_obs)

            episode_rewards.append(ep_ret)
            print(f"[Eval] Episode {ep + 1}/{num_episodes} | Return: {ep_ret:.2f}")

        return {
            "episode_rewards": episode_rewards,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
        }

    def save(self, path: str):
        state = {
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.normalize_reward:
            state['reward_rms'] = {
                'mean': self.reward_rms.mean,
                'var': self.reward_rms.var,
                'count': self.reward_rms.count,
            }
        torch.save(state, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state['policy'])
        if 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
        if self.normalize_reward and 'reward_rms' in state:
            rms = state['reward_rms']
            self.reward_rms.mean = rms['mean']
            self.reward_rms.var = rms['var']
            self.reward_rms.count = rms['count']