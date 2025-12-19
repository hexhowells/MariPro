import os
import threading
from collections import deque

import numpy as np
import torch
import torchvision.transforms as T

from slate import SlateClient
from slate import Agent

import gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from shimmy.openai_gym_compatibility import GymV26CompatibilityV0

from utils import ResetWrapper
from model import ActorCritic


class A2CAgent(Agent):
    def __init__(self, env, model, checkpoint_dir="checkpoints"):
        self.env = env
        self.checkpoint_dir = checkpoint_dir
        self.device = 'cpu'
        self.model = ActorCritic(actions=env.action_space.n).to(self.device)
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(1),
            T.Resize((84, 84)),
            T.ToTensor()
        ])
        
        self.frame_buffer = deque([], maxlen=4)
        self.frame_stack_size = 4
        self.prev_value = 0.0
    

    def preprocess_frame(self, frame):
        frame = frame.squeeze(0).permute(1, 2, 0).numpy()
        frame = (frame * 255).astype(np.uint8)
        frame_tensor = self.transform(frame)

        return frame_tensor.squeeze(0).numpy().astype(np.float32)
    

    def get_action(self, frame):
        processed_frame = self.preprocess_frame(frame)

        self.frame_buffer.append(processed_frame)
        
        while len(self.frame_buffer) < self.frame_stack_size:
            self.frame_buffer.insert(0, processed_frame)
        
        stacked_frames = np.stack(self.frame_buffer, axis=0)  # (4, 84, 84)
        
        obs_t = torch.tensor(stacked_frames, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            logits, value = self.model(obs_t)
            action = torch.argmax(logits, dim=-1).item()
            self.prev_value = float(value.item())
        
        return int(action)
    

    def load_checkpoint(self, checkpoint: str) -> None:
        #checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint)
        checkpoint_path = checkpoint

        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
    

    def get_q_values(self):
        return [self.prev_value]


def create_slate_env():
    env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="rgb_array")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = ResetWrapper(env)
    env = GymV26CompatibilityV0(env=env)

    return env


def start_slate_client(model, checkpoint_dir="checkpoints"):
    def run_client():
        env = create_slate_env()
        agent = A2CAgent(env, model, checkpoint_dir=checkpoint_dir)
        runner = SlateClient(
            env,
            agent,
            endpoint="localhost",
            run_local=True,
            checkpoints_dir=checkpoint_dir,
            frame_rate=0.0
        )
        runner.start_client()
    
    thread = threading.Thread(target=run_client, daemon=True)
    thread.start()
    return thread

