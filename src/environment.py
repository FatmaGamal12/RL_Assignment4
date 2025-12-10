"""
Environment wrapper module for Assignment 4.

This module provides a unified interface for the Box2D continuous-action
environments required in the assignment:

  • LunarLander-v3  (continuous=True)
  • CarRacing-v3    (continuous=True)

We use Gymnasium wrappers for:
  - time limits
  - optional video recording (needed for Hugging Face submission)
"""

from typing import Tuple, Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from gymnasium.wrappers import (
    RecordVideo,
    TimeLimit,
)

# ---------------------------------------------------------
#  CNN Encoder for CarRacing (image → 128-dim feature)
# ---------------------------------------------------------
class CarRacingCNN(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()

        # CNN that processes 96x96x3 images
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # → (32, 23, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → (64, 10, 10)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → (64, 8, 8)
            nn.ReLU(),
        )

        # Compute flatten size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 96, 96)
            conv_out = self.conv_layers(dummy)
            self.flatten_size = conv_out.view(1, -1).shape[1]
            print("CarRacingCNN conv output size:", self.flatten_size)

        # Fully connected layer: flatten → feature_dim
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        # Input could already be (1, 3, 96, 96) from wrapper
        if x.ndim == 3:        # HWC
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.ndim == 4:      # Already BCHW
            pass
        else:
            raise ValueError(f"CarRacingCNN: unexpected obs shape {x.shape}")

        x = x / 255.0
        h = self.conv_layers(x)
        h = h.flatten(start_dim=1)  # keep batch dim
        return self.fc(h)


# ---------------------------------------------------------
#       CarRacing Wrapper (applied inside Environment)
# ---------------------------------------------------------
class CarRacingCNNWrapper(gym.Wrapper):
    """
    Wraps CarRacing so that:
       96x96x3 image  → CNN → 128-d vector
    """

    def __init__(self, env: gym.Env, feature_dim: int = 128):
        super().__init__(env)
        self.feature_dim = feature_dim
        self.cnn = CarRacingCNN(feature_dim).eval()  # no training here

        # Override observation space: now a 1D feature vector
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_dim,),
            dtype=np.float32,
        )

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Convert HWC image → CHW tensor → CNN → 128-dim vector."""
        obs_t = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        obs_t = obs_t / 255.0
        with torch.no_grad():
            features = self.cnn(obs_t).squeeze(0).cpu().numpy()
        return features.astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.encode(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.encode(obs), reward, terminated, truncated, info


# ---------------------------------------------------------
#         Unified Environment Wrapper
# ---------------------------------------------------------
class EnvironmentWrapper:
    """
    Unified wrapper for Box2D continuous environments.

    Usage:
        env = EnvironmentWrapper("LunarLander-v3")
        obs, info = env.reset()
        action = env.sample_action()
        next_obs, reward, terminated, truncated, info = env.step(action)
    """

    SUPPORTED_ENVIRONMENTS = [
        "LunarLander-v3",
        "CarRacing-v3",
    ]

    def __init__(
        self,
        env_name: str,
        render_mode: Optional[str] = None,
        record_video: bool = False,
        max_steps: int = 1000,
        video_dir: str = "videos/",
        # NEW: control whether to use the 128-d CNN encoder for CarRacing
        use_carracing_cnn: bool = True,
    ):
        """
        Initialize the Box2D environment.

        Args:
            env_name: Environment name ("LunarLander-v3" or "CarRacing-v3").
            render_mode: 'human', 'rgb_array', or None.
            record_video: If True, record every episode to video_dir.
            max_steps: Maximum steps per episode (TimeLimit wrapper).
            video_dir: Directory to store recorded videos.
            use_carracing_cnn:
                - True  → CarRacing-v3 is wrapped with CarRacingCNNWrapper
                          (observations are 128-d vectors)   [SAC/TD3 path]
                - False → CarRacing-v3 returns raw 96x96x3 images
                          (for PPO + Nature CNN)
        """
        if env_name not in self.SUPPORTED_ENVIRONMENTS:
            raise ValueError(
                f"Environment {env_name} not supported. "
                f"Supported environments: {self.SUPPORTED_ENVIRONMENTS}"
            )

        self.env_name = env_name
        self.use_carracing_cnn = use_carracing_cnn

        # Decide render mode (for recording we need rgb_array)
        if render_mode is None and record_video:
            render_mode = "rgb_array"

        # --------- CREATE BASE GYM ENVIRONMENT --------- #
        if env_name == "LunarLander-v3":
            self.env = gym.make(
                "LunarLander-v3",
                continuous=True,
                render_mode=render_mode,
            )
        elif env_name == "CarRacing-v3":
            # CarRacing always starts as raw pixel env
            self.env = gym.make(
                "CarRacing-v3",
                continuous=True,
                render_mode=render_mode,
            )

        # 1) Limit episode length
        self.env = TimeLimit(self.env, max_episode_steps=max_steps)

        # 2) If CarRacing and we want shared 128-d encoder → apply CNN wrapper
        self.using_carracing_cnn = False
        if env_name == "CarRacing-v3" and self.use_carracing_cnn:
            self.env = CarRacingCNNWrapper(self.env)
            self.using_carracing_cnn = True

        # 3) Optional video recording
        if record_video:
            # record every episode (episode_trigger=lambda ep: True)
            self.env = RecordVideo(self.env, video_dir, episode_trigger=lambda ep: True)

        # Cache spaces for convenience
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # In this assignment, both envs are continuous (Box spaces)
        self.is_discrete_action = False

    # --------------- STANDARD ENV API --------------- #

    def reset(self, seed: int | None = None) -> Tuple[Any, Dict]:
        """Reset the environment and return (observation, info)."""
        return self.env.reset(seed=seed)

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Take one step in the environment.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        return self.env.step(action)

    def render(self) -> Any:
        """Render the environment."""
        return self.env.render()

    def close(self) -> None:
        """Close the environment."""
        self.env.close()

    # --------------- HELPER METHODS --------------- #

    def get_state_dim(self) -> int:
        """
        Return a "state dimension" for algorithms that expect a flat vector.

        - For LunarLander: observation_space.shape == (8,) → returns 8
        - For CarRacing with CNN wrapper: (128,) → returns 128
        - For raw CarRacing images (96, 96, 3): returns 96*96*3,
          but PPO with Nature CNN should NOT rely on this and instead
          use observation_space.shape directly.
        """
        shape = self.observation_space.shape
        if shape is None:
            raise ValueError("Environment has no observation shape.")
        return int(np.prod(shape))

    def get_action_dim(self) -> int:
        """Return action dimension (continuous)."""
        return int(self.action_space.shape[0])

    def sample_action(self) -> Any:
        """Sample a random action from the action space."""
        return self.action_space.sample()

    def __str__(self) -> str:
        return (
            f"EnvironmentWrapper({self.env_name})\n"
            f"  Observation space: {self.observation_space}\n"
            f"  Action space: {self.action_space}\n"
            f"  Using CarRacing CNN: {self.using_carracing_cnn}\n"
            f"  Discrete actions: {self.is_discrete_action}"
        )


# Convenience factory function (nice for train.py)
def make_env(
    env_name: str,
    render_mode: Optional[str] = None,
    record_video: bool = False,
    max_steps: int = 1000,
    video_dir: str = "videos/",
    use_carracing_cnn: bool = True,
) -> EnvironmentWrapper:
    """
    Convenience function to create an EnvironmentWrapper.

    Examples:
        # LunarLander (unchanged)
        env = make_env("LunarLander-v3")

        # CarRacing for SAC/TD3 (128-d CNN encoder)
        env = make_env("CarRacing-v3", use_carracing_cnn=True)

        # CarRacing for PPO with Nature CNN (raw pixels)
        env = make_env("CarRacing-v3", use_carracing_cnn=False)
    """
    return EnvironmentWrapper(
        env_name=env_name,
        render_mode=render_mode,
        record_video=record_video,
        max_steps=max_steps,
        video_dir=video_dir,
        use_carracing_cnn=use_carracing_cnn,
    )
