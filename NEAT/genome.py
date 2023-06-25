import node_types as types
from node import NodeGene
from connection import ConnectGene
import random
import utils


class Genome:
	""" Class to represent the Genome

		Args:
			sensor_num (int): number of sensor nodes
			output_num (int): number of output nodes
	"""
	def __init__(self, 
			sensor_num, 
			output_num,
			innovation=0,
			coefficient1=1,
			coefficient2=1,
			coefficient3=1):
		self.node_genes = []
		self.connect_genes = []

		self.next_index = 0
		self.innovation = innovation

		self.sensor_num = sensor_num
		self.output_num = output_num

		self.coefficient1 = coefficient1
		self.coefficient2 = coefficient2
		self.coefficient3 = coefficient3
		self.fitness = 0


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


	def add_hidden_nodes(self, n):
		""" Add n hidden nodes to genome
			Used for crossover to initialise an offpring with the correct number of nodes

			Args:
				n (int): number of hidden nodes to create
		"""
		for _ in range(n):
			hidden_node = NodeGene(self.get_next_index(), types.HIDDEN)
			self.node_genes.append(hidden_node)


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
		def accumulate_connections(node, visited):
			value = 0
			for connection in node.connections:
				in_node = self.node_genes[connection.in_node]
				if in_node in visited:
					continue
				else:
					visited.append(in_node)
				if len(in_node.connections) == 0:
					if in_node.ref != None:  # found hidden node without any inbound nodes
						value += x[in_node.ref] * connection.weight
					else:
						value = 0
				else:
					value += accumulate_connections(in_node, visited)

			return utils.sigmoid(value)


		output_nodes = self.node_genes[self.sensor_num : self.sensor_num+self.output_num]
		
		final_output = []
		for out_node in output_nodes:
			final_output.append(accumulate_connections(out_node, []))

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
			for n_in in in_nodes:
				if not self.node_genes[n_out].has_connection(n_in):
					connection = ConnectGene(n_in, n_out, self.innovation, random.uniform(-0.1, 0.1))
					self.connect_genes.append(connection)
					self.node_genes[n_out].add_connection(connection)
					return


	def mutate_weight(self, mutation_rate, random_rate):
		""" Mutate connection weights

			Args:
				mutation_rate (float): rate at which each weight get mutated
				random_rate (float): rate at which mutated weight gets assigned a new random value
		"""
		for con in self.connect_genes:
			if random.random() < mutation_rate:
				if random.random() < random_rate:
					con.weight = random.uniform(-1, 1)
				else:
					con.weight += random.uniform(0.1, -0.1)


	def get_non_matching_genes(self, genome):
		""" Gets all excess and disjoint genes given a comparison genome

			Args:
				genome (Genome): genome to compare against
		"""
		excess_genes = []
		disjoint_genes = []
		for c in self.connect_genes:
			if (c not in genome.connect_genes):
				if (c.innovation > genome.innovation):
					excess_genes.append(c)
				else:
					disjoint_genes.append(c)

		return excess_genes, disjoint_genes


	def get_matching_genes(self, genome):
		""" Get all matching connection genes between two genomes

			Args:
				genome (Genome): genome to get matching genes against
		"""
		matching_genes = []

		for con1 in self.connect_genes:
			for con2 in genome.connect_genes:
				if con1 == con2:
					matching_genes.append((con1, con2))

		return matching_genes


	def compute_distance_score(self, genome):
		""" Compute the distance score between two genomes

			Args:
				genome (Genome): genome to compute the distance score against
		"""
		excess1, disjoint1 = self.get_non_matching_genes(genome)
		excess2, disjoint2 = genome.get_non_matching_genes(self)

		excess = len(excess1) + len(excess2)
		disjoint = len(disjoint1) + len(disjoint2)

		num_genes = max(len(self), len(genome))

		matching = self.get_matching_genes(genome)
		_avg_weight_diff = [abs(x[0].weight - x[1].weight) for x in matching]
		
		if len(_avg_weight_diff) == 0:
			avg_weight_diff = 0
		else:
			avg_weight_diff = sum(_avg_weight_diff) / len(_avg_weight_diff)

		seg1 = (self.coefficient1 * excess) / num_genes
		seg2 = (self.coefficient2 * disjoint) / num_genes
		seg3 = self.coefficient3 * avg_weight_diff

		return seg1 + seg2 + seg3
