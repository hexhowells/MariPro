import torch
from torch import nn


class Layer(nn.Module):
	def __init__(self, in_size, out_size):
		super().__init__()

		self.layer = nn.Linear(in_size, out_size)
		self.activation = nn.Tanh()


	def forward(self, x):
		x = self.layer(x)
		x = self.activation(x)

		return x


class Model(nn.Module):
	def __init__(self, dims):
		super().__init__()

		layers = [Layer(d1, d2) for d1, d2 in zip(dims, dims[1:])]
		self.layers = nn.Sequential(*layers)


	def forward(self, x):
		x = x.flatten()
		x = self.layers(x)

		return x


class ConvModel(nn.Module):
	def __init__(self, dims):
		super().__init__()

		self.features = nn.Sequential(
					nn.Conv2d(1, 4, kernel_size=2, bias=False),
					nn.ReLU(inplace=True),
					nn.Conv2d(4, 8, kernel_size=2, bias=False),
					nn.ReLU(inplace=True)
				)

		self.linear = nn.Sequential(
					nn.Linear(1232, 6),
					nn.ReLU(inplace=True)
				)
		

	def forward(self, x):
		# add batch and channel dims
		x = x.unsqueeze(0).unsqueeze(0)

		x = self.features(x)

		x = x.flatten()
		x = self.linear(x)

		return x
