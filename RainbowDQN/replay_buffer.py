from collections import deque
import random
import torch
import numpy as np


class ReplayBuffer:
	def __init__(self, capacity, frame_stack, transform):
		self.frame_stack = frame_stack
		self.capacity = capacity + frame_stack
		self.transform = transform

		self.frames = torch.zeros((self.capacity, 84, 84), dtype=torch.float)
		self.actions = torch.zeros(self.capacity, dtype=torch.int64)
		self.rewards = torch.zeros(self.capacity, dtype=torch.int8)
		self.dones = torch.ones(self.capacity, dtype=torch.bool)

		self.p = 0  # pointer to next availble slot
		self.size = 0  # number of filled slots


	def append(self, frame, action, reward, done):
		self.frames[self.p] = self.transform(frame)
		self.actions[self.p] = action
		self.rewards[self.p] = reward
		self.dones[self.p] = done

		self.p = (self.p + 1) % self.capacity
		self.size = max(self.size, self.p)


	def add_first_frame(self, frame):
		self.frames[self.p] = self.transform(frame)
		self.actions[self.p] = 0
		self.rewards[self.p] = 0
		self.dones[self.p] = False

		self.p = (self.p + 1) % self.capacity
		self.size = max(self.size, self.p)


	def stack_frames(self, index):
		frame_stack = torch.zeros((4, 84, 84), dtype=torch.float)

		for p in range(self.frame_stack-1, -1, -1):
			frame_stack[p] = self.frames[index]
			if self.dones[(index - 1) % self.capacity] == False:
				index -= 1

		return frame_stack


	def current_state(self):
		return self.stack_frames(self.p-1).unsqueeze(0)


	def _sample_indices(self, batch_size):
		indices = set()

		while len(indices) < batch_size:
			index = random.randint(0, self.size-1)
			if self.dones[(index - 1) % self.capacity] != True:
				indices.add(index)

		return list(indices)


	def sample(self, batch_size):
		indices = self._sample_indices(batch_size)

		states = torch.stack([self.stack_frames(i - 1) for i in indices]).to('cuda')
		actions = self.actions[indices].unsqueeze(1).to('cuda')
		rewards = self.rewards[indices].to('cuda')
		dones = self.dones[indices].to('cuda')
		next_states = torch.stack([self.stack_frames(i) for i in indices]).to('cuda')

		return states, actions, rewards, next_states, dones


	def __len__(self):
		return self.size


class PrioritisedReplayBuffer:
	def __init__(
			self, 
			capacity, 
			frame_stack, 
			transform,
			n_step=3,
			gamma=0.99
		):
		self.frame_stack = frame_stack
		self.capacity = capacity + frame_stack
		self.transform = transform

		self.frames = torch.zeros((self.capacity, 84, 84), dtype=torch.float)
		self.actions = torch.zeros(self.capacity, dtype=torch.int64)
		self.rewards = torch.zeros(self.capacity, dtype=torch.int8)
		self.dones = torch.ones(self.capacity, dtype=torch.bool)
		self.priorities = torch.zeros(self.capacity, dtype=torch.float)  # p_i^alpha

		self.p = 0  # pointer to next availble slot
		self.size = 0  # number of filled slots
		self.alpha = 0.6
		self.beta = 0.4
		self.epsilon = 1e-6

		self.n_step = n_step
		self.gamma = gamma
		self.n_step_buffer = deque(maxlen=n_step)



	def _get_n_step_transition(self):
		n_step_reward = 0

		for i, (_, _, reward, _) in enumerate(self.n_step_buffer):
			n_step_reward += (self.gamma ** i) * reward

		state_index, action, _, _ = self.n_step_buffer[0]
		_, _, _, done = self.n_step_buffer[-1]
		_, _, _, next_index = self.n_step_buffer[-1]

		return state_index, action, n_step_reward, done, next_index



	def append(self, frame, action, reward, done, priority):
		index = self.p

		self.frames[index] = self.transform(frame)
		self.actions[index] = action
		self.rewards[index] = reward
		self.dones[index] = done
		self.priorities[index] = (abs(priority) + self.epsilon) ** self.alpha

		self.n_step_buffer.append((index, action, reward, done))

		if len(self.n_step_buffer) == self.n_step or done:
			state_index, action, n_step_reward, done_flag, next_index = self._get_n_step_transition()

			# go back and update older transition slots now that N steps ahead have been captured
			self.actions[state_index] = action
			self.rewards[state_index] = n_step_reward
			self.dones[state_index] = done_flag

		self.p = (self.p + 1) % self.capacity
		self.size = max(self.size, self.p)

		if done:
			self.n_step_buffer.clear()


	def add_first_frame(self, frame):
		self.frames[self.p] = self.transform(frame)
		self.actions[self.p] = 0
		self.rewards[self.p] = 0
		self.dones[self.p] = False
		self.priorities[self.p] = 0.0

		self.p = (self.p + 1) % self.capacity
		self.size = max(self.size, self.p)


	def stack_frames(self, index):
		frame_stack = torch.zeros((4, 84, 84), dtype=torch.float)

		for p in range(self.frame_stack-1, -1, -1):
			frame_stack[p] = self.frames[index]
			if self.dones[(index - 1) % self.capacity] == False:
				index -= 1

		return frame_stack


	def current_state(self):
		return self.stack_frames(self.p-1).unsqueeze(0)


	def compute_probabilities(self):
		probs = self.priorities[:self.size]
		return (probs / probs.sum()).numpy()


	def _sample_indices(self, probs, batch_size):
		return np.random.choice(self.size, batch_size, replace=False, p=probs)


	def update_priorities(self, indices, priorities):
		self.priorities[indices] = priorities
		

	def sample(self, batch_size):
		probs = self.compute_probabilities()
		indices = self._sample_indices(probs, batch_size)

		states = torch.stack([self.stack_frames(i - 1) for i in indices]).to('cuda')
		actions = self.actions[indices].unsqueeze(1).to('cuda')
		rewards = self.rewards[indices].to('cuda')
		dones = self.dones[indices].to('cuda')
		next_states = torch.stack([self.stack_frames((i + self.n_step - 1) % self.capacity) for i in indices]).to('cuda')

		weights = (self.size * torch.tensor(probs[indices])).pow(-self.beta)
		weights = (weights / weights.max()).to('cuda')

		return states, actions, rewards, next_states, dones, weights, indices


	def __len__(self):
		return self.size