import os
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from skimage.transform import resize

# Correct VizDoom imports
from vizdoom.vizdoom import DoomGame, ScreenFormat

class ViZDoomEnv(Env):
    def __init__(self):
        self.game = DoomGame()

        self.config_path = os.path.abspath("../configs/basic.cfg")
        self.scenario_path = os.path.abspath("../scenarios/basic.wad")

        self.game.load_config(self.config_path)
        self.game.set_doom_scenario_path(self.scenario_path)
        self.game.set_window_visible(True)
        self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.init()

        self.observation_space = Box(low=0, high=255, shape=(12, 84, 84), dtype=np.uint8)

        self.actions = [
            [1, 0, 0], 
            [0, 1, 0],  
            [0, 0, 1],  
            [1, 0, 1],  
            [0, 1, 1],  
            [0, 0, 0]
        ]

      
        self.action_space = Discrete(len(self.actions))

    def step(self, action):
        reward = self.game.make_action(self.actions[action])
        done = self.game.is_episode_finished()
        obs = self._get_obs()

        if action == 2 and not self._enemy_in_view():
            reward -= 1.0

        return obs, float(reward), done, False, {}

    def reset(self, seed=None, options=None):
        self.game.new_episode()
        return self._get_obs(), {}

    def _get_obs(self):
        if self.game.is_episode_finished():
            return np.zeros((3, 84, 84), dtype=np.uint8)

        obs = self.game.get_state().screen_buffer  # shape (240, 320, 3)
        obs = np.transpose(obs, (2, 0, 1))          # (3, 240, 320)
        obs = resize(obs, (12, 84, 84), anti_aliasing=True)

        return (obs * 255).astype(np.uint8)

    def _enemy_in_view(self):
        if self.game.get_state():
            obs = self._get_obs()
            center_region = obs[:, 100:140, 150:170]
            return np.std(center_region) > 20
        return False

    def render(self):
        return self._get_obs()

    def close(self):
        self.game.close()
