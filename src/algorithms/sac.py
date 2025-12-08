import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple
import random
from collections import deque


# ============================
#  ACTOR NETWORK
# ============================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        # outputs mean and log_std for Gaussian policy
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)   # limit stability

        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()

        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True), mean


# ============================
#  CRITIC NETWORK
# ============================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


# ============================
#  SAC MAIN CLASS
# ============================
class SAC:

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = config

        lr_actor  = config["learning_rate_actor"]
        lr_critic = config["learning_rate_critic"]

        # Actor
        self.actor = Actor(state_dim, action_dim)

        # Critic 1 & 2
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)

        # Target critics
        self.target1 = Critic(state_dim, action_dim)
        self.target2 = Critic(state_dim, action_dim)

        # copy weights
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        # ----------------- DEVICE -----------------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.move_to_device()

        # Optimizers
        self.actor_opt   = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)


        # Entropy temperature α
        self.alpha = config["alpha"]
        self.automatic_entropy_tuning = config["automatic_entropy_tuning"]

        if self.automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.tensor(
                np.log(self.alpha),device=self.device,requires_grad=True,
            )
            self.alpha_opt = torch.optim.Adam(
                [self.log_alpha],lr=config.get("learning_rate_alpha", self.cfg["learning_rate_actor"]))

        # Replay buffer
        self.replay = deque(maxlen=config["replay_memory_size"])

        self.gamma = config["discount_factor"]
        self.tau = config["tau"]
        self.batch_size = config["batch_size"]

    def move_to_device(self):
        self.actor.to(self.device)
        self.critic1.to(self.device)
        self.critic2.to(self.device)
        self.target1.to(self.device)
        self.target2.to(self.device)

    # =====================================================
    #  Select action
    # =====================================================
    def select_action(self, state: np.ndarray, deterministic=False, env_name=None):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # ======================= Deterministic (for evaluation) =======================
        if deterministic:
            with torch.no_grad():
                mean, _ = self.actor(state)
                action = torch.tanh(mean).cpu().numpy()[0]
        else:
            # ======================= Stochastic sampling (training) =======================
            with torch.no_grad():
                action, _, _ = self.actor.sample(state)  
                action = action.cpu().numpy()[0]

        # ======================= Action Fix for CarRacing-v3 =======================
        # Steering  ∈ [-1,1]
        # Gas      ∈ [0,1]
        # Brake    ∈ [0,1]
        if env_name == "CarRacing-v3":  # Only apply when env is CarRacing
            steering = action[0]                 # unchanged
            gas      = (action[1] + 1) / 2        # scale from [-1,1] → [0,1]
            brake    = (action[2] + 1) / 2        # scale from [-1,1] → [0,1]
            action = np.array([steering, gas, brake], dtype=np.float32)

        return action


    # =====================================================
    #  Replay buffer store
    # =====================================================
    def store(self, transition):
        self.replay.append(transition)

    def sample_batch(self):
        batch = random.sample(self.replay, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(rewards).unsqueeze(1).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
            torch.FloatTensor(dones).unsqueeze(1).to(self.device),
        )

    # =====================================================
    #  Soft update
    # =====================================================
    def soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    # =====================================================
    #  Training loop
    # =====================================================
    def train(self, env, config=None, logger=None):
        episodes = self.cfg["episodes"]
        total_rewards = []
         # convergence parameters
        # Optional convergence parameters
        convergence_threshold = self.cfg.get("convergence_threshold", None)
        convergence_window = self.cfg.get("convergence_window", 50)

        for ep in range(episodes):
            state, _ = env.reset()
            ep_reward = 0
            done = False

            while not done:
                action = self.select_action(
                    state,
                    deterministic=False,
                    env_name=getattr(env, "env_name", None),
                )
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.store((state, action, reward, next_state, done))
                state = next_state
                ep_reward += reward

                if len(self.replay) > self.batch_size:
                    self.update_networks()

            total_rewards.append(ep_reward)

            # Running average for convergence
            window = min(len(total_rewards), convergence_window)
            avg_reward = np.mean(total_rewards[-window:])

            print(f"Episode {ep+1}/{episodes} | Reward={ep_reward:.2f} | Avg({window})={avg_reward:.2f}")

            if logger:
                logger.log({"reward": ep_reward, "avg_reward": avg_reward})

            # # ====================== CHECK EARLY CONVERGENCE ======================
            # if convergence_threshold is not None and len(total_rewards) >= convergence_window:
            #     if avg_reward >= convergence_threshold:
            #         print("\n================= SAC CONVERGED =================")
            #         print(f"Average reward {avg_reward:.2f} over last {convergence_window} eps")
            #         print(f"Training stopped early at episode {ep+1}")
            #         print("=================================================\n")
            #         break
                
        mean_reward = np.mean(total_rewards)
        print("\n========= TRAINING COMPLETE =========")
        print(f" Mean Reward = {mean_reward:.2f}")
        print("=====================================\n")

        return {
            "mean_reward": mean_reward,
            "max_reward": float(np.max(total_rewards)),
            "min_reward": float(np.min(total_rewards)),
        }

    # =====================================================
    #  SAC update step
    # =====================================================
    def update_networks(self):
        states, actions, rewards, next_states, dones = self.sample_batch()

        with torch.no_grad():
            next_action, next_logprob, _ = self.actor.sample(next_states)

            q1_next = self.target1(next_states, next_action)
            q2_next = self.target2(next_states, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_logprob

            target_q = rewards + (1 - dones) * self.gamma * q_next

        # Critic updates
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        # Actor update
        action_new, logprob_new, _ = self.actor.sample(states)

        q1_new = self.critic1(states, action_new)
        q2_new = self.critic2(states, action_new)
        q_min = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * logprob_new - q_min).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Entropy temperature tuning
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (logprob_new + self.target_entropy).detach()).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            self.alpha = self.log_alpha.exp()

        # soft update
        self.soft_update(self.target1, self.critic1)
        self.soft_update(self.target2, self.critic2)
        

    # =====================================================
    #  Testing
    # =====================================================
    def test(self, env, num_episodes=10):
        results = []
        for ep in range(num_episodes):
            state, _ = env.reset()
            done = False
            ep_reward = 0

            while not done:
                action = self.select_action(state,deterministic=True,env_name=getattr(env, "env_name", None),)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                state = next_state

            results.append(ep_reward)
            print(f"[TEST] Episode {ep+1}/{num_episodes} | Reward: {ep_reward:.2f}")

        return {
            "mean_reward": float(np.mean(results)),
            "std_reward": float(np.std(results)),
            "min_reward": float(np.min(results)),
            "max_reward": float(np.max(results)),
            "episode_rewards": results,
        }

    # =====================================================
    #  Save / Load
    # =====================================================
    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
