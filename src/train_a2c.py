# a2c_train.py

from environment import ViZDoomEnv
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.spaces import Box
from skimage.transform import resize
import gymnasium as gym
import numpy as np
import os

# Custom Resize Wrapper using skimage
class ResizeImage(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = Box(low=0, high=255, shape=(self.shape[0], self.shape[1], 3), dtype=np.uint8)

    def observation(self, observation):
        resized = resize(observation, (self.shape[0], self.shape[1], 3), anti_aliasing=True)
        return (resized * 255).astype(np.uint8)

# Create environment with preprocessing
def make_env():
    env = ViZDoomEnv()                       # Your custom ViZDoom environment
    env = ResizeImage(env, shape=(84, 84))   # Resize frames to 84x84
    return env

# Wrap environment
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

# A2C Model Setup
model = A2C(
    "CnnPolicy",
    env,
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    gae_lambda=1.0,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_rms_prop=True,
    verbose=1,
    policy_kwargs={"net_arch": [64, 64]}  # Small CNN + MLP policy
)

# Start Training
model.learn(total_timesteps=40_000)

# Save Model
os.makedirs("models", exist_ok=True)
model.save("models/a2c2_vizdoom")

# Clean Up
env.close()
