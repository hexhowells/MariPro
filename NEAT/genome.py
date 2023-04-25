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
		ref_str = "Ref: "+str(self.ref) if self.ref is not None else ""
		return f'NodeGene {self.idx}\tType: {self.type}\t{ref_str}'


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

		Args:
			sensor_num (int): number of sensor nodes
			output_num (int): number of output nodes
	"""
	def __init__(self, sensor_num, output_num):
		self.node_genes = []
		self.connect_genes = []
		self.next_index = 0
		self.innovation = 0
		self.sensor_num = sensor_num
		self.output_num = output_num


	def __len__(self):
		return len(self.node_genes)


	def get_next_index(self):
		idx = self.next_index
		self.next_index += 1
		return idx


	def initialise_nodes(self):
		""" Initialise the node genes in the Genome
			Will create nodes for all output and input nodes possible
		"""

		# add sensor nodes
		for i in range(self.sensor_num):
			sensor_node = NodeGene(self.get_next_index(), types.SENSOR, i)
			self.node_genes.append(sensor_node)

		# add output nodes
		for i in range(self.output_num):
			out_node = NodeGene(self.get_next_index(), types.OUTPUT, i)
			self.node_genes.append(out_node)


	def initialise_connections(self, n):
		""" Initialise the connection genes in the Genome
			Will create random connections between existing genes

			Args:
				n (int): number of connections to create
		"""
		assert len(self) != 0, "Need to initialise the Node Genes before initialising the Connection Genes"

		a = 0
		b = self.sensor_num
		c = self.sensor_num + 1
		d = self.sensor_num + self.output_num
		connection_tuples = [(random.randint(a, b), random.randint(c, d)) for _ in range(n)]

		for (in_node, out_node) in connection_tuples:
			connection = ConnectGene(in_node, out_node, self.innovation, random.uniform(-0.1, 0.1))
			self.connect_genes.append(connection)
			


	def forward(self, x):
		pass
		# x is the input array flattened
		# create a list of all values in x that are being used as sensor nodes
		# create a list for all hidden and output nodes, initialisating with all zeros
		# for each sensor node
		#   find all outgoing connections
		#   multiply sensor node with connection weight and accumulate value in respective out node
		#   repeat above two lines but for the out node 
		# (forward breadth-first search)
		#
		# for each output node
		#   find all incoming connections
		#   if the in_node has incoming connections then recursively find their incoming connections
		#   if the in_node has no incoming connections then multiply the node with the connection
		#      and store result in parent node
		# (reverse depth-first search)


	def mutate_node(self):
		pass
		# pick a random connection from the gene pool
		# add a new node and two new connections using the info from the old connection
		# and conforming to the mutation rules (see paper)
		# delete old connection?


	def mutate_connection(self):
		pass
		# find two unconnected nodes, add a random connection between them


	def get_excess_nodes(self, connections):
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
