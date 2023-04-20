import node_types as types
import random


class NodeGene:
	""" Class to represent the Node Gene

		Args:
			node_idx (int): index of the node
			node_type (string): node type
			ref (int): reference index to input or output array
	"""
	def __init__(self, node_idx, node_type, ref=None):
		self.idx = node_idx
		self.type = node_type
		self.ref = ref

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
		self.sensor_nodes = []
		self.hidden_nodes = []
		self.output_nodes = []
		self.connect_genes = []
		self.next_index = 0
		self.innovation = 0


	def __len__(self):
		return len(sensor_nodes) + len(hidden_nodes) + len(output_nodes)


	def get_next_index(self):
		idx = self.next_index
		self.next_index += 1
		return idx


	def get_node_genes(self):
		return self.sensor_nodes + self.hidden_nodes + self.output_nodes


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
			out_node = NodeGene(self.get_next_index(), types.OUTPUT)
			self.output_nodes.append(out_node)

		# add random sensor nodes
		random_indexes = random.sample(range(0, max_sensor_idx), n)

		for idx in random_indexes:
			sensor_node = NodeGene(self.get_next_index(), types.SENSOR)
			self.sensor_nodes.append(sensor_node)


	def initialise_connections(self, n):
		""" Initialise the connection genes in the Genome
			Will create random connections between existing genes

			Args:
				n (int): number of connections to create
		"""
		assert len(self) != 0, "Need to initialise the Node Genes before initialising the Connection Genes"

		for _ in range(n):
			pass


	def forward(self, x):
		pass
		# x is the input array flattened
		# 


	def mutate_node(self):
		pass
		# pick a random connection from the gene pool
		# add a new node and two new connections using the info from the old connection
		# and conforming to the mutation rules (see paper)
		# delete old connection?


	def mutate_connection(self):
		pass
		# find two unconnected nodes, add a random connection between them


	def get_excess_nodes(self):
		pass
		# if connection exists in self but not in connections
		# and if self.connection.innovation is above max innovation number in connections
		# then connection is excess


	def get_disjoint_nodes(self, connections):
		pass
		# if connection exists in self but not in connections
		# and if self.connection.innovation is below max innovation number in connections
		# then connection is disjoint


	def compute_distance_score(self, genome):
		pass
		# get excess and disjoint nodes from self using genome.connections
		# get avg weight difference between self.connections and genome.connections
		# get max( len(self), len(genome) )
