import numpy as np
import random
import torch


def layer_crossover(evo):
    """ layer crossover, alternates between selecting the layer+bias from each parent

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
        # choose two parents to breed from
        parent_idxs = np.random.choice(range(len(evo.population)), size=2, replace=False, p=probs)

        a_layers = list(evo.population[parent_idxs[0]].parameters())
        b_layers = list(evo.population[parent_idxs[1]].parameters())

        offspring_chromosome = evo.model(evo.dims)

        use_parent_a = True

        # apply crossover
        for i in range(0, len(a_layers), 2):
            if use_parent_a:
                list(offspring_chromosome.parameters())[i] = a_layers[i]
                list(offspring_chromosome.parameters())[i+1] = a_layers[i+1]
            else:
                list(offspring_chromosome.parameters())[i] = b_layers[i]
                list(offspring_chromosome.parameters())[i+1] = b_layers[i+1]

            use_parent_a = not use_parent_a


        offspring.append(offspring_chromosome)

        avg_fitness = (evo.fitness_scores[parent_idxs[0]] + evo.fitness_scores[parent_idxs[1]]) // 2
        fitness.append(avg_fitness)

    return offspring, fitness


def average_crossover(evo):
    """ average crossover, averages the weights of both parents together

        Args:
            evo (Evolution): Evolution class instance
    """
    probs = np.asarray(evo.fitness_scores) / sum(evo.fitness_scores)

    # determine how many chromosomes to create
    original_pop_size = len(evo.population) / evo.survival_rate
    size = int( original_pop_size - len(evo.population) )

    offspring, fitness = [], []

    for _ in range(size):
        # choose two parents to breed from
        parent_idxs = np.random.choice(range(len(evo.population)), size=2, replace=False, p=probs)

        # apply crossover
        sdA = evo.population[parent_idxs[0]].state_dict()
        sdB = evo.population[parent_idxs[1]].state_dict()

        # average all parameters
        for key in sdA:
            sdB[key] = (sdB[key] + sdA[key]) / 2.

        # recreate model and load averaged state_dict (or use modelA/B)
        offspring_chromosome = evo.model(evo.dims)
        offspring_chromosome.load_state_dict(sdB)
        offspring.append(offspring_chromosome)
        
        # set new members fitness to 0 since it hasnt been evaluated
        avg_fitness = (evo.fitness_scores[parent_idxs[0]] + evo.fitness_scores[parent_idxs[1]]) // 2
        fitness.append(avg_fitness)

    return offspring, fitness