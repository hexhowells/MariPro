import gym

from gymnasium.wrappers import TransformObservation
from gymnasium import wrappers
from gymnasium.spaces import Box

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from shimmy.openai_gym_compatibility import GymV26CompatibilityV0

import torch
import torchvision.transforms as T

import numpy as np
from PIL import Image
import time


class ResetWrapper:
    """Wrapper to ignore seed parameter in reset() for environments that don't support it"""
    def __init__(self, env):
        self.env = env
    
    def reset(self, seed=None, options=None):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
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
        env = ResetWrapper(env)

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


def render_policy(model, env_id="SuperMarioBros-v0", max_steps=1000, seed=123):
    env = gym.make(env_id, apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = ResetWrapper(env)
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
    
    obs, _ = env.reset()

    device = next(model.parameters()).device
    total_reward = 0.0

    for _ in range(max_steps):
        time.sleep(0.01)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
            action = torch.argmax(logits, dim=-1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    env.close()

    return total_reward