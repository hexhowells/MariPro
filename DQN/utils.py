import hyperparameters as hp
import random
import torch
from frame_buffer import FrameBuffer
import numpy as np



def epsilon_greedy(policy_net, frame, epsilon):
    if random.random() < epsilon:
        action = random.randint(0, hp.action_space-1)
    else:
        with torch.no_grad():
            q_values = policy_net(frame.to(device='cuda'))
            action = torch.argmax(q_values).item()
    
    return action


def create_minibatch(replay_buffer, policy_net, target_net):
    # sample from replay buffer
    transitions = random.sample(replay_buffer, hp.batch_size)

    # extract from minibatch transitions
    frames = torch.stack([t[0].squeeze(0) for t in transitions]).to('cuda')
    actions = torch.LongTensor([t[1] for t in transitions]).unsqueeze(1).to('cuda')
    rewards = torch.FloatTensor([t[2] for t in transitions]).to('cuda')
    next_frames = torch.stack([t[3].squeeze(0) for t in transitions]).to('cuda')
    dones = torch.BoolTensor([t[4] for t in transitions]).to('cuda')

    # Double DQN logic
    next_q_actions = policy_net(next_frames).argmax(dim=1, keepdim=True)
    next_q_values = target_net(next_frames).gather(1, next_q_actions).squeeze(1)

    # compute target
    targets = torch.where(dones, torch.tensor(-1.0, device='cuda'), rewards + hp.gamma * next_q_values)

    return frames, actions, targets


def evaluate(env, policy, transform, n_episodes=5):
    rewards = []
    
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            fb = FrameBuffer(obs, hp.history_len, transform)
            done, ep_r = False, 0

            while not done:
                a = epsilon_greedy(policy, fb.state(), epsilon=0.05)
                obs, r, done = env.step(a)
                fb.append(obs)
                ep_r += r
            rewards.append(ep_r)
    
    return np.mean(rewards)

