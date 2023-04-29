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

	__call__ = forward


	def mutate_node(self):
		""" Mutate node
			selecs a random connection, disables it, then creates a new node with two new connections
			joining the two nodes that were previously connected.
		"""
		self.innovation += 1
		
		rand_connection_idx = random.randint(0, len(self.connect_genes)-1)
		rand_connection = self.connect_genes[rand_connection_idx]
		rand_connection.enabled = False

		node_idx = self.get_next_index()
		new_node = NodeGene(node_idx, types.HIDDEN)
		self.node_genes.append(new_node)

		# old in node -> new node
		new_connection1 = ConnectGene(rand_connection.in_node, node_idx, self.innovation, weight=1)
		self.connect_genes.append(new_connection1)
		self.node_genes[node_idx].add_connection(new_connection1)

		# new node -> old out node
		new_connection2 = ConnectGene(node_idx, rand_connection.out_node, self.innovation, weight=rand_connection.weight)
		self.connect_genes.append(new_connection2)
		self.node_genes[rand_connection.out_node].add_connection(new_connection2)



	def mutate_connection(self):
		""" Mutate connection
			creates a new connection between two unconnected nodes
			Doesnt connect sensor nodes to sensor nodes or output nodes to output nodes
		"""
		self.innovation += 1

		out_nodes = list(range(self.sensor_num, len(self.node_genes)))
		in_nodes = list(range(0, self.sensor_num)) + list(range(self.sensor_num+self.output_num, len(self.node_genes)))

		random.shuffle(out_nodes)
		random.shuffle(in_nodes)

		for n_out in out_nodes:
			out_connections = self.node_genes[n_out].connections
			for n_in in in_nodes:
				if n_in not in out_connections:
					connection = ConnectGene(n_in, n_out, self.innovation, random.uniform(-0.1, 0.1))
					self.connect_genes.append(connection)
					self.node_genes[n_out].add_connection(connection)
					return


	def get_non_matching_genes(self, connections, max_innov):
		""" Gets all excess and disjoint genes given a comparison genome

			Args:
				connections (list): list of connections from comparison genome
				max_innov (int): max innovation number used in connections
		"""
		excess_genes = []
		disjoint_genes = []
		for c in self.connect_genes:
			if (c not in connections):
				if (c.innovation > max_innov):
					excess_genes.append(c)
				else:
					disjoint_genes.append(c)

		return excess_genes, disjoint_genes


	def compute_distance_score(self, genome):
		pass
		# get excess and disjoint nodes from self using genome.connections
		# get avg weight difference between self.connections and genome.connections
		# get max( len(self), len(genome) )
