import random

class ConnectGene:
	""" Class to represent the Connection Gene

		Args:
			in_node (int): index for the input node
			out_node (int): index for the output node
			innovatinon (int): the innovation number
			weight (float): the connection weight
			enabled (boolean): if the connection is enabled
	"""
	def __init__(self, in_node, out_node, innovation, weight=None, enabled=True):
		self.in_node = in_node
		self.out_node = out_node
		self.innovation = innovation
		self.weight = weight if weight else random.uniform(-1, 1)
		self.enabled = enabled

	def __str__(self):
		return f'ConnectGene {self.in_node} -> {self.out_node}\tInnovation: {self.innovation}\tenabled={self.enabled}'


	def __eq__(self, obj):
		return (self.in_node == obj.in_node) and (self.out_node == obj.out_node)