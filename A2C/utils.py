import gym
from gymnasium.wrappers import TransformObservation
from gymnasium import wrappers
from gymnasium.spaces import Box

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from shimmy.openai_gym_compatibility import GymV26CompatibilityV0

import torchvision.transforms as T

import numpy as np


class EnvWrapper:
    def __init__(self, env):
        self.env = env
    
    def reset(self, seed=None, options=None):
        return self.env.reset()
    
    def step(self, action):
        result = self.env.step(action)
        obs, reward, terminated, truncated, info = result
        reward = np.clip(reward, -1.0, 1.0)
        return obs, reward, terminated, truncated, info
    
    def render(self, mode=None):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    def __getattr__(self, name):
        return getattr(self.env, name)


def make_env(env_id: str, seed: int):
    def thunk():
        env = gym.make(env_id, apply_api_compatibility=True, render_mode="rgb_array")
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = EnvWrapper(env)

        env = GymV26CompatibilityV0(env=env)

        transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(1),
            T.Resize((84, 84)),
            T.ToTensor()
        ])
        
        def preprocess_frame(obs):
            obs = np.array(obs, dtype=np.uint8)
            frame_tensor = transform(obs)

            return frame_tensor.squeeze(0).numpy().astype(np.float32)
        
        obs_space_pre = Box(
            low=0.0,
            high=1.0,
            shape=(84, 84),
            dtype=np.float32
        )
        env = TransformObservation(env, preprocess_frame, obs_space_pre)
        env = wrappers.FrameStackObservation(env, stack_size=4)

        return env
    
    return thunk
