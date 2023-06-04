from genome import Genome
import random

def crossover(parent1, parent2, fitness1, fitness2):
	matching_genes = parent1.get_matching_genes(parent2)

	connect_genes = [random.choice([c1, c2]) for (c1, c2) in matching_genes]

	if fitness1 > fitness2:
		excess_genes, disjoint_genes = parent1.get_non_matching_genes(parent2)
		max_innov = parent1.innovation
		node_len = len(parent1.node_genes)
	else:
		excess_genes, disjoint_genes = parent2.get_non_matching_genes(parent1)
		max_innov = parent2.innovation
		node_len = len(parent2.node_genes)

	for dc in disjoint_genes:
		connect_genes.append(dc)
	
	for ec in excess_genes:
		connect_genes.append(ec)

	offspring = Genome(parent1.sensor_num, parent1.output_num, innovation=max_innov)
	offspring.initialise_nodes()

	offspring.connect_genes = connect_genes

	offspring.add_hidden_nodes(node_len - len(offspring.node_genes))

	return offspring
