import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, TransformObservation
from gymnasium import wrappers
from gymnasium.spaces import Box
import torch
import numpy as np
import socket


def make_env(env_id: str, seed: int):
    def thunk():
        env = gym.make(env_id, frameskip=1)
        env.reset(seed=seed)

        env = AtariPreprocessing(
            env,
            frame_skip=4,
            grayscale_obs=True,
            scale_obs=False,
            terminal_on_life_loss=False,
        )

        env = wrappers.FrameStackObservation(env, stack_size=4)

        obs_space = Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32
        )
        
        env = TransformObservation(
            env, 
            lambda x: np.array(x, dtype=np.float32) / 255.0,
            obs_space)

        return env
    return thunk


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 1))
    ip = s.getsockname()[0]
    s.close()

    return ip
