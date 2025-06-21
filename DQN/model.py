import torch
from torch import nn


class QNetwork(nn.Module):
	def __init__(self, in_channels=4, actions=7):
		super().__init__()

		self.features = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=2, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1),
			nn.ReLU(inplace=True),
			)
		self.fc = nn.Sequential(
			nn.Flatten(),
			nn.Linear(512, 100),
			nn.Linear(100, actions),
			)


	def forward(self, x):
		x = self.features(x)
		x = self.fc(x)

		return x
