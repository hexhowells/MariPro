import os

import torch
import torchvision.transforms as T

from slate import Agent
from model import ActorCritic


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
