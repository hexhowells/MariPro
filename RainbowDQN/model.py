import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class NoisyLinear(nn.Module):
	def __init__(self, in_features, out_features):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features

		# Learnable parameters
		self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
		self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
		self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

		self.bias_mu = nn.Parameter(torch.empty(out_features))
		self.bias_sigma = nn.Parameter(torch.empty(out_features))
		self.register_buffer("bias_epsilon", torch.empty(out_features))

		self.reset_parameters()
		self.reset_noise()


	def reset_parameters(self):
		mu_range = 1 / np.sqrt(self.in_features)

		self.weight_mu.data.uniform_(-mu_range, mu_range)
		self.weight_sigma.data.fill_(0.017)
		self.bias_mu.data.uniform_(-mu_range, mu_range)
		self.bias_sigma.data.fill_(0.017)


	def reset_noise(self):
		self.weight_epsilon.normal_()
		self.bias_epsilon.normal_()


	def forward(self, x):
		if self.training:
			weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
			bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
		else:
			weight = self.weight_mu
			bias = self.bias_mu

		return F.linear(x, weight, bias)


class DuelingDQN51(nn.Module):
	def __init__(
			self, 
			in_channels=4, 
			actions=7,
			atom_size=51,
			v_min=-10,
			v_max=10
		):
		super().__init__()

		self.atom_size = atom_size
		self.v_min = v_min
		self.v_max = v_max
		self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to('cuda')
		self.actions = actions

		self.features = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=(8,8), stride=4),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=(4,4), stride=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=(3,3), stride=1),
			nn.ReLU(inplace=True),
			)

		self.flatten = nn.Flatten()

		self.adv_fc1 = NoisyLinear(3136, 512)
		self.adv_fc2 = NoisyLinear(512, actions * atom_size)

		self.val_fc1 = NoisyLinear(3136, 512)
		self.val_fc2 = NoisyLinear(512, atom_size)


	def forward(self, x):
		x = self.features(x)
		x = self.flatten(x)

		a_values = F.relu(self.adv_fc1(x))
		a_values = self.adv_fc2(a_values).view(-1, self.actions, self.atom_size)

		s_value = F.relu(self.val_fc1(x))
		s_value = self.val_fc2(s_value).view(-1, 1, self.atom_size)

		q_atoms = s_value + (a_values - a_values.mean(dim=1, keepdim=True))
		probabilities = torch.softmax(q_atoms, dim=2)

		return probabilities


	def q_values(self, x):
		probs = self.forward(x)
		q_vals = torch.sum(probs * self.support, dim=2)

		return q_vals


	def reset_noise(self):
		self.adv_fc1.reset_noise()
		self.adv_fc2.reset_noise()
		self.val_fc1.reset_noise()
		self.val_fc2.reset_noise()
