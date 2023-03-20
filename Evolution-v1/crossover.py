import numpy as np
import random


def uniform_crossover(evo):
    """ Uniform crossover

        Args:
            evo (Evolution): Evolution class instance
    """
    # probability distribution based from the fitness scores
    probs = np.asarray(evo.fitness_scores) / sum(evo.fitness_scores)

    # determine how many chromosomes to create
    original_pop_size = len(evo.population) / evo.survival_rate
    size = int( original_pop_size - len(evo.population) )

    offspring, fitness = [], []

    for _ in range(size):
        # determines which parent to take each gene from (parent 0 or parent 1)
        bitmask = random.choices([0, 1], k=evo.chromosome_length)

        # choose two parents to breed from
        parent_idxs = np.random.choice(range(len(evo.population)), size=2, replace=False, p=probs)

        # apply crossover
        offspring.append( [evo.population[parent_idxs[p]][gene] for gene, p in enumerate(bitmask)] )

        avg_fitness = (evo.fitness_scores[parent_idxs[0]] + evo.fitness_scores[parent_idxs[1]]) // 2
        fitness.append(avg_fitness)

    return offspring, fitness


def one_point_crossover(evo):
    """ One-point crossover

        Args:
            evo (Evolution): Evolution class instance
    """
    # probability distribution based from the fitness scores
    probs = np.asarray(evo.fitness_scores) / sum(evo.fitness_scores)

    # determine how many chromosomes to create
    original_pop_size = len(evo.population) / evo.survival_rate
    size = int( original_pop_size - len(evo.population) )

    offspring, fitness = [], []

    for _ in range(size):
        crossover_point = random.choice(range(1, evo.chromosome_length-1))

        # choose two parents to breed from
        parent_idxs = np.random.choice(range(len(evo.population)), size=2, replace=False, p=probs)

        # apply crossover
        chromosome_parent_a = evo.population[parent_idxs[0]][:crossover_point]
        chromosome_parent_b = evo.population[parent_idxs[1]][crossover_point:]
        offspring.append( chromosome_parent_a + chromosome_parent_b )

        avg_fitness = (evo.fitness_scores[parent_idxs[0]] + evo.fitness_scores[parent_idxs[1]]) // 2
        fitness.append(avg_fitness)

    return offspring, fitness


def k_point_crossover(evo, k=5):
    """ K-point crossover

        Args:
            evo (Evolution): Evolution class instance
            k (int): number of crossover points
    """
    # probability distribution based from the fitness scores
    probs = np.asarray(evo.fitness_scores) / sum(evo.fitness_scores)

    # determine how many chromosomes to create
    original_pop_size = len(evo.population) / evo.survival_rate
    size = int( original_pop_size - len(evo.population) )

    offspring, fitness = [], []

    for _ in range(size):
        crossover_points = random.sample(range(1, evo.chromosome_length-1), k=k)
        crossover_points.append(0)
        crossover_points.append(evo.chromosome_length)
        crossover_points = sorted(crossover_points)

        # choose two parents to breed from
        parent_idxs = np.random.choice(range(len(evo.population)), size=2, replace=False, p=probs)

        offspring_chromosome = []

        # apply crossover
        for i in range(len(crossover_points)-1):
            p_idx = parent_idxs[(i % 2)]
            offspring_chromosome += evo.population[p_idx][crossover_points[i] : crossover_points[i+1]]

        offspring.append( offspring_chromosome )

        avg_fitness = (evo.fitness_scores[parent_idxs[0]] + evo.fitness_scores[parent_idxs[1]]) // 2
        fitness.append(avg_fitness)

    return offspring, fitness
