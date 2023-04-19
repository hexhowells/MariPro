import node_types as types
import random


class NodeGene:
	""" Class to represent the Node Gene

		Args:
			node_idx (int): index of the node
			node_type (string): node type
	"""
	def __init__(self, node_idx, node_type):
		self.idx = node_idx
		self.type = node_type

	def __str__(self):
		return f'NodeGene {self.idx}, Type: {self.type}'


class ConnectGene:
	""" Class to represent the Connection Gene

		Args:
			node_in (int): index for the input node
			node_out (int): index for the output node
			innovatinon (int): the innovation number
			weight (float): the connection weight
			enabled (boolean): if the connection is enabled
	"""
	def __init__(self, node_in, node_out, innovation, weight=None, enabled=True):
		self.node_in = node_in
		self.node_out = node_out
		self.innovation = innovation
		self.weight = weight if weight else random.uniform(-1, 1)
		self.enabled = enabled


class Genome:
	""" Class to represent the Genome
	"""
	def __init__(self):
		self.node_genes = []
		self.connect_genes = []


	def initialise_nodes(self, n, max_sensor_idx, max_output_idx):
		""" Initialise the node genes in the Genome
			Will create nodes for all outputs but a random set of nodes for the inputs

			Args:
				n (int): number of input nodes to create
				max_sensor_idx (int): number of sensor nodes
				max_output_idx (int): number of output nodes
		"""
		# add output nodes
		for i in range(max_output_idx):
			out_node = NodeGene(i, types.OUTPUT)
			self.node_genes.append(out_node)

		# add random sensor nodes
		random_indexes = random.sample(range(0, max_sensor_idx), n)

		for idx in random_indexes:
			sensor_node = NodeGene(idx, types.SENSOR)
			self.node_genes.append(sensor_node)


		
