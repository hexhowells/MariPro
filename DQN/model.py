import torch
from torch import nn


class QNetwork(nn.Module):
	def __init__(self, in_channels=12, actions=7):
		super().__init__()

		self.features = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=(8,8), stride=4),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=(4,4), stride=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=(3,3), stride=1),
			nn.ReLU(inplace=True),
			)
		self.fc = nn.Sequential(
			nn.Flatten(),
			nn.Linear(3136, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, actions),
			)


	def forward(self, x):
		x = self.features(x)
		x = self.fc(x)

		return x
