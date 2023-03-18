import numpy as np


def _sort_population(evo):
	""" Sort the population by fitness scores
		Returns the population and fitness scores (sorted and paired)

		Args:
			evo (Evolution): Evolution class instance
	"""
	population_data = [(evo.population[i], evo.fitness_scores[i]) for i in range(len(evo.population))]
	population_data = sorted(population_data, key=lambda chromosome: chromosome[1], reverse=True)

	population, fitness_scores = zip(*population_data)

	return list(population), list(fitness_scores)


def roulette_wheel_selection(evo):
	""" Roulette wheel selection function

		Args:
			evo (Evolution): Evolution class instance
	"""
	probs = np.asarray(evo.fitness_scores) / sum(evo.fitness_scores)

	population_size = len(evo.population)
	survivor_size = int(population_size * evo.survival_rate)
	indexes = np.random.choice(range(population_size), size=survivor_size, replace=False, p=probs)

	evo.population = [evo.population[i] for i in indexes]
	evo.fitness_scores = [evo.fitness_scores[i] for i in indexes]


def elitist_selection(evo, topk=3):
	""" Roulette wheel selection function with elitist selection

		Args:
			evo (Evolution): Evolution class instance
			topk (int): how many top individuals guaranteed to be selected
	"""
	population, fitness_scores = _sort_population(evo)

	# keep topk performing chromosomes
	top_population = population[:topk]
	top_fitness_scores = fitness_scores[:topk]

	# remove kept chromosomes
	population = population[topk:]
	fitness_scores = fitness_scores[topk:]

	probs = np.asarray(fitness_scores) / sum(fitness_scores)

	survivor_size = int(len(evo.population) * evo.survival_rate) - topk
	indexes = np.random.choice(range(len(population)), size=survivor_size, replace=False, p=probs)

	evo.population = top_population + [population[i] for i in indexes]
	evo.fitness_scores = top_fitness_scores + [fitness_scores[i] for i in indexes]


def rank_selection(evo):
	""" Rank selection function

		Args:
			evo (Evolution): Evolution class instance
	"""
	pass

	# sort the population accoring to the fitness scores
	# assign probabilities based on their rank
	# methods
	#   1. use their position and create a prob distribution
	#
	#   2. use the equation P(i) = (C - c * (i - 1)) / N (linear ranking)
	#       C = normalisation constant
	#       c = selection pressure
	#       N = population size
	#       i = rank index
	#
	#   3. P(i) = (1 - exp(-c * i)) / (1 - exp(-c * N)) (exponential ranking)


def tournament_selection(evo):
	""" Tournament selection function

		Args:
			evo (Evolution): Evolution class instance
	"""
	pass


def truncation_selection(evo):
	""" Truncation selection function

		Args:
			evo (Evolution): Evolution class instance
	"""
	pass
	# select the top n individuals