from environment import ViZDoomEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from skimage.transform import resize
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import os

# Resize Wrapper using Skimage
class ResizeImage(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        # Set observation space correctly and ignore Pyright warning
        self.observation_space = Box(low=0, high=255, shape=(self.shape[0], self.shape[1], 3), dtype=np.uint8)  # type: ignore

    def observation(self, observation):
        # Resize and convert to uint8 (as required by SB3)
        resized = resize(observation, (self.shape[0], self.shape[1], 3), anti_aliasing=True)
        return (resized * 255).astype(np.uint8)

# Create Environment
def make_env():
    env = ViZDoomEnv()                       # Load your ViZDoom environment
    env = ResizeImage(env, shape=(84, 84))   # Apply resizing
    return env

# Wrap with DummyVecEnv and FrameStack (VecTransposeImage not needed if channels-last)
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

# PPO Model Setup
model = PPO(
    "CnnPolicy",
    env,
    n_steps=2048,
    batch_size=64,
    learning_rate=2.5e-4,
    gamma=0.99,
    ent_coef=0.01,
    clip_range=0.2,
    verbose=1,
    policy_kwargs={
        "net_arch": [64, 64],  # Custom small architecture
    }
)

# Start Training
model.learn(total_timesteps=30_000)

# Save Model
os.makedirs("models", exist_ok=True)
model.save("models/ppo_vizdoom")

# Clean Up
env.close()
