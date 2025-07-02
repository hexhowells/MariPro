from collections import deque
import random


class ReplayBuffer:
	def __init__(self, size):
		self.buffer = deque(maxlen=size)


	def __len__(self):
		return len(self.buffer)


	def append(self, transition):
		self.buffer.append(transition)


	def sample(self, batch_size):
		return random.sample(self.buffer, batch_size)



class PriorityExperienceReplay:
	def __init__(self):
		pass


	def __len__(self):
		pass


	def append(self):
		pass


	def sample(self):
		pass