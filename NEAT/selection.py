import numpy as np
import math
import random
import copy


def _sort_population(population, fitness_scores):
	""" Sort the population by fitness scores
		Returns the population and fitness scores (sorted and paired)

		Args:
			population (list): list of genomes
			fitness_scores (list): list of fitness scores for each genome
	"""
	population_data = [(population[i], fitness_scores[i]) for i in range(len(population))]
	population_data = sorted(population_data, key=lambda chromosome: chromosome[1], reverse=True)

	population, fitness_scores = zip(*population_data)

	return list(population), list(fitness_scores)


def roulette_wheel_selection(population, fitness_scores, survival_rate):
	""" Roulette wheel selection function

		Args:
			population (list): list of genomes
			fitness_scores (list): list of fitness scores for each genome
			survival_rate (float): number of survivors in the population
	"""
	probs = np.asarray(fitness_scores) / sum(fitness_scores)

	population_size = len(population)
	survivor_size = int(population_size * survival_rate)
	indexes = np.random.choice(range(population_size), size=survivor_size, replace=False, p=probs)

	new_population = [population[i] for i in indexes]
	new_fitness_scores = [fitness_scores[i] for i in indexes]

	return new_population


def elitist_selection(population, fitness_scores, survival_rate, topk=3):
	""" Roulette wheel selection function with elitist selection

		Args:
			population (list): list of genomes
			fitness_scores (list): list of fitness scores for each genome
			survival_rate (float): number of survivors in the population
			topk (int): how many top individuals guaranteed to be selected
	"""
	population, fitness_scores = _sort_population(population, fitness_scores)

	# keep topk performing chromosomes
	top_population = population[:topk]
	top_fitness_scores = fitness_scores[:topk]

	# remove kept chromosomes
	population = population[topk:]
	fitness_scores = fitness_scores[topk:]

	probs = np.asarray(fitness_scores) / sum(fitness_scores)

	survivor_size = int(len(population) * survival_rate) - topk
	indexes = np.random.choice(range(len(population)), size=survivor_size, replace=False, p=probs)

	population = top_population + [population[i] for i in indexes]
	fitness_scores = top_fitness_scores + [fitness_scores[i] for i in indexes]

	return population


def linear_rank_selection(population, fitness_scores, survival_rate, sp=1.9):
	""" Linear rank selection function

		Args:
			population (list): list of genomes
			fitness_scores (list): list of fitness scores for each genome
			survival_rate (float): number of survivors in the population
			sp (float): selection pressure (1.0 = no selection pressure, 2.0 = high selection pressure)
	"""
	population, fitness_scores = _sort_population(population, fitness_scores)
	probs = []
	n = len(population)

	#probs = [1/n * (sp - (2 * sp - 2) * ((i-1) / (n-1))) for i in range(1, n+1)]

	for i in range(1, n+1):
		p = 1/n * (sp - (2 * sp - 2) * ((i-1) / (n-1)))
		probs.append(p)

	survivor_size = int(n * survival_rate)
	indexes = np.random.choice(range(n), size=survivor_size, replace=False, p=probs)

	population = [population[i] for i in indexes]
	fitness_scores = [fitness_scores[i] for i in indexes]

	return population


def exponential_rank_selection(population, fitness_scores, survival_rate, sp=1.9):
	""" Exponential rank selection function

		Args:
			population (list): list of genomes
			fitness_scores (list): list of fitness scores for each genome
			survival_rate (float): number of survivors in the population
			sp (float): selection pressure (1.0 = no selection pressure, 2.0 = high selection pressure)
	"""
	population, fitness_scores = _sort_population(population, fitness_scores)
	probs = []
	n = len(population)

	for i in range(1, n+1):
		#p = (1 - math.exp(-sp * i)) / (1 - math.exp(-sp * n))
		p = ((sp - 1) / (sp**n - 1)) * sp**(n-i)
		#p = (1 - math.exp(-i)) / sp
		#p = math.exp(-sp * i) / n
		probs.append(p)

	survivor_size = int(n * survival_rate)
	indexes = np.random.choice(range(n), size=survivor_size, replace=False, p=probs)

	population = [population[i] for i in indexes]
	fitness_scores = [fitness_scores[i] for i in indexes]

	return population


def tournament_selection(population, fitness_scores, survival_rate):
	""" Tournament selection function

		Args:
			population (list): list of genomes
			fitness_scores (list): list of fitness scores for each genome
			survival_rate (float): number of survivors in the population
	"""
	k = int(len(population) * survival_rate)
	new_population = []
	new_fitness_scores = []

	# run tournament
	for _ in range(k):
		idx1, idx2 = random.sample(range(len(population)), 2)
		score1 = fitness_scores[idx1]
		score2 = fitness_scores[idx2]
		winner = None

		if score1 > score2:
			winner = idx1
		elif score2 > score1:
			winner = idx2
		else:
			winner = random.sample([idx1, idx2], 1)[0]

		new_population.append(population.pop(winner))
		new_fitness_scores.append(fitness_scores.pop(winner))

	return new_population


def truncation_selection(population, fitness_scores, survival_rate):
	""" Truncation selection function

		Args:
			population (list): list of genomes
			fitness_scores (list): list of fitness scores for each genome
			survival_rate (float): number of survivors in the population
	"""
	population, fitness_scores = _sort_population(population, fitness_scores)
	survivor_size = int(len(population) * survival_rate)

	population = population[:survivor_size]
	fitness_scores = fitness_scores[:survivor_size]

	return population