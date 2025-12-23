import argparse
import ale_py
import gymnasium as gym
import numpy as np
from slate import SlateClient
from slate import Agent

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, TransformObservation
from gymnasium import wrappers
from gymnasium.spaces import Box
import numpy as np
import torch
import os

from model import ActorCritic


def make_slate_env():
    env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode='rgb_array')
    env.reset()

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


class SlateAgentICM(Agent):
    def __init__(self, env, checkpoint_dir):
        self.env = env
        self.checkpoint_dir = checkpoint_dir
        self.device = 'cpu'
        self.model = ActorCritic(actions=env.action_space.n).to(self.device)
        self.value = None
        
    def get_action(self, frame):
        obs_t = torch.tensor(frame, dtype=torch.float32, device=self.device)
        obs_t = obs_t.unsqueeze(0)
        logits, self.value = self.model(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        
        return int(action.item())
    
    def load_checkpoint(self, checkpoint: str) -> None:
        if os.path.exists(checkpoint):
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
            self.model.eval()
            print(f"Loaded checkpoint: {checkpoint}")
        else:
            print(f"Checkpoint not found: {checkpoint}")

    def get_q_values(self):
        return [self.value]
