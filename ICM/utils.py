import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, TransformObservation
from gymnasium import wrappers
from gymnasium.spaces import Box
import torch
import numpy as np


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


def render_policy(model, env_id, seed=123):
    env = gym.make(env_id, frameskip=1, render_mode="human")
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
    
    obs, _ = env.reset(seed=seed)

    device = next(model.parameters()).device
    total_reward = 0.0
    last_score_steps = 0

    for i in range(100_000_000):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
            action = torch.argmax(logits, dim=-1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if reward > 0: last_score_steps = i

        if terminated or truncated or (i - last_score_steps) > 500:
            break

    env.close()

    return total_reward