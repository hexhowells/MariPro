import node_types as types
from node import NodeGene
from connection import ConnectGene
import random


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
		return len(self.node_genes) + len(self.connect_genes)


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
		b = self.sensor_num - 1
		c = self.sensor_num
		d = self.sensor_num + self.output_num - 1
		connection_tuples = [(random.randint(a, b), random.randint(c, d)) for _ in range(n)]

		for (in_node, out_node) in connection_tuples:
			connection = ConnectGene(in_node, out_node, self.innovation, random.uniform(-0.1, 0.1))
			self.connect_genes.append(connection)
			print(out_node)
			self.node_genes[out_node].add_connection(connection)
			


	def forward(self, x):
		""" Compute forward pass through the network

			Args:
				x (array): array/list storing the values of the inputs
		"""
		def accumulate_connections(node):
			value = 0
			for connection in node.connections:
				in_node = self.node_genes[connection.in_node]
				if len(in_node.connections) == 0:
					value += x[in_node.ref] * connection.weight
				else:
					value += accumulate_connections(in_node)

			return value


		output_nodes = self.node_genes[self.sensor_num : self.sensor_num+self.output_num]
		
		final_output = []
		for out_node in output_nodes:
			final_output.append(accumulate_connections(out_node))

		return final_output


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
