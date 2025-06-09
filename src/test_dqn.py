import numpy as np
from stable_baselines3 import DQN
from environment import ViZDoomEnv
from skimage.transform import resize
import gymnasium as gym
from gymnasium.spaces import Box

# Resize wrapper
class ResizeImage(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = Box(
            low=0, high=255, shape=(self.shape[0], self.shape[1], 3), dtype=np.uint8
        )

    def observation(self, obs):
        resized = resize(obs, (self.shape[0], self.shape[1], 3), anti_aliasing=True)
        return (resized * 255).astype(np.uint8)

# Make environment
def make_env():
    env = ViZDoomEnv()
    env = ResizeImage(env, shape=(84, 84))
    return env

# Load environment
env = make_env()

# Load trained DQN model
model = DQN.load("models/dqn_vizdoom.zip")  # Change filename if using a different seed

# Run test episodes
num_episodes = 30

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Handle reward types safely
        if isinstance(reward, dict):
            total_reward += float(list(reward.values())[0])
        elif isinstance(reward, (np.ndarray, list, tuple)):
            total_reward += float(reward[0])
        else:
            total_reward += float(reward)

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
