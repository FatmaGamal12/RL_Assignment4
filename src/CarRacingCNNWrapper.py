import gymnasium as gym  
import torch
import torch.nn as nn
import numpy as np


class CarRacingCNNWrapper(gym.Wrapper):
    def __init__(self, env, device="cpu", feature_dim=128):
        super().__init__(env)
        self.device = device
        self.feature_dim = feature_dim

        # CNN backbone (for 96x96x3 observations)
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),   # -> (32, 23, 23)
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),  # -> (64, 10, 10)
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),  # -> (64, 8, 8)
            nn.Flatten(),                               # -> 64*8*8 = 4096
        )

        # Linear head to get final feature vector
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, feature_dim),  # ✅ 64*8*8, not 64*10*10
            nn.ReLU()
        )

        # IMPORTANT: tell Gym the new observation shape (feature_dim,)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_dim,),
            dtype=np.float32,
        )

    def preprocess(self, obs):
        # obs: (96, 96, 3) uint8
        obs = obs.transpose(2, 0, 1) / 255.0  # -> (3, 96, 96), in [0,1]
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z = self.cnn_backbone(obs)
            z = self.fc(z)
        return z.cpu().numpy()[0]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.preprocess(obs), info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        return self.preprocess(next_obs), reward, terminated, truncated, info
