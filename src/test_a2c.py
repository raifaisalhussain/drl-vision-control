import os
import numpy as np
from stable_baselines3 import A2C
from environment import ViZDoomEnv  # Make sure this points to your actual environment file

# Load the trained model
model_path = os.path.abspath("models/a2c2_vizdoom.zip")
model = A2C.load(model_path)

# Create the environment
env = ViZDoomEnv()

# Run test episodes
num_episodes = 30

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        # FIXED: Handle reward type safely
        if isinstance(reward, dict):
            reward_val = list(reward.values())[0]
        elif isinstance(reward, (list, tuple, np.ndarray)):
            reward_val = reward[0]
        else:
            reward_val = reward

        total_reward += float(reward_val)

        env.render()

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
