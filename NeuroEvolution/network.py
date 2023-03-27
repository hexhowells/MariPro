import torch
from torch import nn


class Layer(nn.Module):
	def __init__(self, in_size, out_size):
		super().__init__()

		self.layer = nn.Linear(in_size, out_size)
		self.activation = nn.ReLU(inplace=True)


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
