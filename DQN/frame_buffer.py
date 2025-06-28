import torch
from collections import deque


class FrameBuffer:
	def __init__(self, frame, buffer_len, transform):
		self.transform = transform

		frame_tensor = self.transform(frame).unsqueeze(0)
		self.frame_stack = deque([frame_tensor] * buffer_len, maxlen=buffer_len)


	def append(self, frame):
		frame_tensor = self.transform(frame).unsqueeze(0)
		self.frame_stack.append(frame_tensor)


	def stack_frames(self):
		return torch.cat(list(self.frame_stack), dim=1)


	def state(self):
		return self.stack_frames()


