import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from environment import ViZDoomEnv  # Your custom environment
from gymnasium.spaces import Box
from skimage.transform import resize
import numpy as np

# Resize Wrapper to preprocess observation images
class ResizeImage(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = Box(low=0, high=255, shape=(self.shape[0], self.shape[1], 3), dtype=np.uint8)

    def observation(self, observation):
        resized = resize(observation, (self.shape[0], self.shape[1], 3), anti_aliasing=True)
        return (resized * 255).astype(np.uint8)

# Function to create and wrap environment
def make_env():
    env = ViZDoomEnv()
    env = ResizeImage(env, shape=(84, 84))
    return env

# Train a DQN model once with total_timesteps
def train_dqn():
    os.makedirs("models", exist_ok=True)

    # Wrap the environment
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)  # Important for CNN input shape

    # Initialize the model
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        seed=0,
        verbose=1,
    )

    # Train the model
    model.learn(total_timesteps=30_000)

    # Save the model
    model.save("models/dqn_vizdoom")
    print("Model saved.")

    # Close the environment
    env.close()

# Run training only once
if __name__ == "__main__":
    train_dqn()
