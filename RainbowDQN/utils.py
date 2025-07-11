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


def compute_targets(policy_net, target_net, transitions):
    _, _, rewards, next_frames, dones = transitions

    # Double DQN logic
    with torch.no_grad():
       next_q_actions = policy_net(next_frames).argmax(dim=1, keepdim=True)
    next_q_values = target_net(next_frames).gather(1, next_q_actions).squeeze(1)

    # DQN logic
    #next_q_values = target_net(next_frames).argmax(dim=1, keepdim=True).squeeze(1)

    # compute target
    targets = torch.where(dones, torch.tensor(-1.0, device='cuda'), rewards + hp.gamma * next_q_values)

    return targets


def evaluate(env, policy, transform, n_episodes=5):
    rewards = []
    best_dist = 0

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            frame_buffer = FrameBuffer(obs, hp.history_len, transform)
            done, ep_reward = False, 0

            while not done:
                action = epsilon_greedy(policy, frame_buffer.state(), epsilon=0.05)
                obs, reward, done = env.step(action)
                
                frame_buffer.append(obs)
                ep_reward += reward

            rewards.append(ep_reward)

        best_dist = max(best_dist, env.high_score)
    
    return np.mean(rewards), best_dist

