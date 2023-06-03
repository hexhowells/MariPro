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
		self.connections = []

	def add_connection(self, connection):
		self.connections.append(connection)

	def has_connection(self, in_node):
		for connection in self.connections:
			if in_node == connection.in_node:
				return True
		return False

	def __str__(self):
		ref_str = "Ref: "+str(self.ref) if self.ref is not None else ""
		return f'NodeGene {self.idx}\tType: {self.type}\t{ref_str}'