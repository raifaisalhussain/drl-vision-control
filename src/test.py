import os
from stable_baselines3 import PPO
from environment import ViZDoomEnv

# Update path to match actual model file location
model_path = os.path.abspath("models/ppo_vizdoom4.zip")  # or "src/ppo_vizdoom"

model = PPO.load(model_path)
env = ViZDoomEnv()

# Running the model for some episodes
num_episodes = 30
for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        # Ensure the reward is a float before adding it
        total_reward += float(reward)

        env.render()

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
