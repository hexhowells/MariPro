import numpy as np
import random
import torch



def uniform_mutation(evo, chromosomes, fitness_scores):
    """ Uniform mutation - mutates random genes uniformly

        Args:
            evo (Evolution): Evolution class instance
            chromosomes (list): chromosomes to mutate
            fitness_scores (list): fitness scores of the chromosomes
    """
    for chromosome in chromosomes:
        for param in chromosome.parameters():
            # Get the number of weights in the linear layer.
            num_weights = param.numel()

            # Compute the number of weights to mutate.
            num_to_mutate = int(num_weights * evo.mutation_rate)

            # Get the indices of the weights to mutate.
            indices = random.sample(range(num_weights), num_to_mutate)

            # Mutate the selected weights.
            with torch.no_grad():
                param.flatten()[indices] += torch.randn(num_to_mutate) * 0.01

        evo.population.append(chromosome)
        evo.fitness_scores.append(0)
