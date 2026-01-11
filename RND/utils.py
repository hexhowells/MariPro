import gym
from gymnasium.wrappers import TransformObservation
from gymnasium import wrappers
from gymnasium.spaces import Box
import gymnasium as gymnasium_lib

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from shimmy.openai_gym_compatibility import GymV26CompatibilityV0

import torchvision.transforms as T
import torch

import numpy as np
import socket


class SkipFrame(gymnasium_lib.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


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
        env = SkipFrame(env, skip=4)

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


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 1))
    ip = s.getsockname()[0]
    s.close()

    return ip


class RunningMeanStd:
    """Tracks running mean and standard deviation for normalization."""
    def __init__(self, epsilon=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon


    def update(self, x):
        x_np = x.detach().cpu().numpy()
        
        x_flat = x_np.flatten()
        batch_mean = float(np.mean(x_flat))
        batch_var = float(np.var(x_flat))
        batch_count = len(x_flat)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count


    def normalize(self, x):
        x_np = x.detach().cpu().numpy()
        
        std = np.sqrt(self.var + 1e-8)
        normalized = (x_np - self.mean) / std

        return normalized
