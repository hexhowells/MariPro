import hyperparameters as hp
import random
import torch
from frame_buffer import FrameBuffer
import numpy as np



def select_action(policy_net, frame):
	with torch.no_grad():
		probs = policy_net(frame.to(device='cuda'))
		q_values = torch.sum(probs * policy_net.support, dim=2)
		action = torch.argmax(q_values, dim=1).item()
	
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


def projection_distribution(next_dist, rewards, dones, gamma, support, v_min, v_max, atom_size):
	batch_size = rewards.size(0)
	delta_z = (v_max - v_min) / (atom_size - 1)
	projected_dist = torch.zeros((batch_size, atom_size)).to('cuda')

	for i in range(atom_size):
		tz = rewards + gamma * support[i] * (1 - dones.float())
		tz = tz.clamp(v_min, v_max)
		b = (tz - v_min) / delta_z
		l = b.floor().long()
		u = b.ceil().long()

		eq_mask = (u == l).float()
		projected_dist[range(batch_size), l] += next_dist[:, i] * (eq_mask + (1 - eq_mask) * (u.float() - b))
		projected_dist[range(batch_size), u] += next_dist[:, i] * (1 - eq_mask) * (b - l.float())

	return projected_dist



def evaluate(env, policy, transform, n_episodes=5):
	rewards = []
	best_dist = 0

	with torch.no_grad():
		for _ in range(n_episodes):
			obs, _ = env.reset()
			frame_buffer = FrameBuffer(obs, hp.history_len, transform)
			done, ep_reward = False, 0

			while not done:
				action = select_action(policy, frame_buffer.state())
				obs, reward, done = env.step(action)
				
				frame_buffer.append(obs)
				ep_reward += reward

			rewards.append(ep_reward)

		best_dist = max(best_dist, env.high_score)
	
	return np.mean(rewards), best_dist

