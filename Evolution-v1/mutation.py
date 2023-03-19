import numpy as np
import random



def uniform_mutation(evo, chromosomes, fitness_scores):
    """ Uniform mutation - mutates random genes uniformly

        Args:
            evo (Evolution): Evolution class instance
            chromosomes (list): chromosomes to mutate
            fitness_scores (list): fitness scores of the chromosomes
    """
    for chromosome in chromosomes:
        number_of_mutations = int(evo.chromosome_length * evo.mutation_rate)
        mutations = [random.randint(0, evo.gene_length-1) for _ in range(number_of_mutations)]

        genes_to_mutate = random.sample(range(evo.chromosome_length), number_of_mutations)

        for gene, mutation in zip(genes_to_mutate, mutations):
            chromosome[gene] = mutation

        evo.population.append(chromosome)
        evo.fitness_scores.append(0)


def local_mutation(evo, chromosomes, fitness_scores, spread_rate=0.1):
    """ Local mutation - only mutate the genes near where the agent stopped

        Args:
            evo (Evolution): Evolution class instance
            chromosomes (list): chromosomes to mutate
            fitness_scores (list): fitness scores of the chromosomes
            spread_rate (float): percentage of genes away from agent stop point to potentially mutate
    """
    spread = int(evo.chromosome_length * spread_rate)

    for chromosome, fitness_score in zip(chromosomes, fitness_scores):
        steps = fitness_score // 5

        number_of_mutations = int(2 * spread * evo.mutation_rate)
        mutations = [random.randint(0, evo.gene_length-1) for _ in range(number_of_mutations)]

        min_gene = max(0, steps - spread)
        max_gene = min(evo.chromosome_length, steps + spread)
        genes_to_mutate = random.sample(range(min_gene, max_gene), number_of_mutations)

        for gene, mutation in zip(genes_to_mutate, mutations):
            chromosome[gene] = mutation

        evo.population.append(chromosome)
        evo.fitness_scores.append(0)


def _create_gaussian(size, n, sigma):
    """ Creates a gaussian distribution

        Args:
            size (int): size of samples in the distribution
            n (int): center of the distribution
            sigma (sigma): standard deviation of the gaussian
    """
    x = np.arange(size)
    prob_distribution = np.exp(-((x - n) ** 2) / (2 * sigma ** 2))
    
    return prob_distribution / np.sum(prob_distribution)


def weighted_mutation(evo, chromosomes, fitness_scores, sigma=125):
    """ Weighted mutation - genes nearest where the agent stopped have a higher probability of being mutated
            Creates a gaussian distribution from the step where the agent stopped to sample from

        Args:
            evo (Evolution): Evolution class instance
            chromosomes (list): chromosomes to mutate
            fitness_scores (list): fitness scores of the chromosomes
            sigma (sigma): standard deviation for the gaussian
    """
    for chromosome, fitness_score in zip(chromosomes, fitness_scores):
        number_of_mutations = int(evo.chromosome_length * evo.mutation_rate)
        mutations = [random.randint(0, evo.gene_length-1) for _ in range(number_of_mutations)]

        # create gaussian distribution centered where the agent terminated
        steps = fitness_score // 5

        gaussian = _create_gaussian(evo.chromosome_length, steps, sigma=sigma)
        genes_to_mutate = np.random.choice(range(evo.chromosome_length), size=number_of_mutations, p=gaussian)

        for gene, mutation in zip(genes_to_mutate, mutations):
            chromosome[gene] = mutation

        evo.population.append(chromosome)
        evo.fitness_scores.append(0)
