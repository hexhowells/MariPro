import numpy as np
import math
import random


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


def linear_rank_selection(evo, sp=1.9):
	""" Linear rank selection function

		Args:
			evo (Evolution): Evolution class instance
			sp (float): selection pressure (1.0 = no selection pressure, 2.0 = high selection pressure)
	"""
	population, fitness_scores = _sort_population(evo)
	probs = []
	n = len(evo.population)

	#probs = [1/n * (sp - (2 * sp - 2) * ((i-1) / (n-1))) for i in range(1, n+1)]

	for i in range(1, n+1):
		p = 1/n * (sp - (2 * sp - 2) * ((i-1) / (n-1)))
		probs.append(p)

	survivor_size = int(n * evo.survival_rate)
	indexes = np.random.choice(range(n), size=survivor_size, replace=False, p=probs)

	evo.population = [evo.population[i] for i in indexes]
	evo.fitness_scores = [evo.fitness_scores[i] for i in indexes]


def exponential_rank_selection(evo, sp=1.9):
	""" Exponential rank selection function

		Args:
			evo (Evolution): Evolution class instance
			sp (float): selection pressure (1.0 = no selection pressure, 2.0 = high selection pressure)
	"""
	population, fitness_scores = _sort_population(evo)
	probs = []
	n = len(evo.population)

	for i in range(1, n+1):
		#p = (1 - math.exp(-sp * i)) / (1 - math.exp(-sp * n))
		p = ((sp - 1) / (sp**n - 1)) * sp**(n-i)
		#p = (1 - math.exp(-i)) / sp
		#p = math.exp(-sp * i) / n
		probs.append(p)

	survivor_size = int(n * evo.survival_rate)
	indexes = np.random.choice(range(n), size=survivor_size, replace=False, p=probs)

	evo.population = [evo.population[i] for i in indexes]
	evo.fitness_scores = [evo.fitness_scores[i] for i in indexes]


def tournament_selection(evo):
	""" Tournament selection function

		Args:
			evo (Evolution): Evolution class instance
	"""
	k = int(len(evo.population) * evo.survival_rate)
	population = []
	fitness_scores = []

	# run tournament
	for _ in range(k):
		idx1, idx2 = random.sample(range(len(evo.population)), 2)
		score1 = evo.fitness_scores[idx1]
		score2 = evo.fitness_scores[idx2]
		winner = None

		if score1 > score2:
			winner = idx1
		elif score2 > score1:
			winner = idx2
		else:
			winner = random.sample([idx1, idx2], 1)

		population.append(evo.population.pop(winner))
		fitness_scores.append(evo.fitness_scores.pop(winner))

	evo.population = population
	evo.fitness_scores = fitness_scores


def truncation_selection(evo):
	""" Truncation selection function

		Args:
			evo (Evolution): Evolution class instance
	"""
	population, fitness_scores = _sort_population(evo)
	survivor_size = int(len(evo.population) * evo.survival_rate)

	evo.population = population[:survivor_size]
	evo.fitness_scores = fitness_scores[:survivor_size]