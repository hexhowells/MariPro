from genome import Genome
import numpy as np
from model import NEAT
import selection
import node_types as types
import utils


def compute_variance(values):
    mean = sum(values) / len(values)
    squared_diffs = [(x - mean) ** 2 for x in values]
    variance = sum(squared_diffs) / len(values)
    return int(variance)


def main():
    # setup environment
    print("\nInitialising Environments...")
    model = NEAT(
    	selection=selection.elitist_selection,
        population_size=300,
		env_name="SuperMarioBros-v0",
		weight_mutation_rate=0.8,
		weight_random_rate=0.1,
		gene_disabled_rate=0.4,
        weight_disable_rate=0.6,
		crossover_rate=0.75,
		new_node_rate=0.5,
		new_link_rate=0.25,
        dist_threshold=0.4,
		coefficient1=2.0,
		coefficient2=0.4,
		coefficient3=1.0
		)

    model.initialise_population()
    model.initialise_species()

    # hyperparameters
    best_fitness = 0
    generations = 100

    # begin simulation
    for generation in range(generations):
        print("-" * 90)
        print(f"\nGeneration: {generation} / {generations}")
        model.simulate_generation()

        fitness_scores = [round(genome.fitness, 2) for genome in model.get_population()]
        num_connections = [sum([1 if con.enabled else 0 for con in genome.connect_genes]) for genome in model.get_population()]
        num_nodes = [sum([1 if (node.type == types.HIDDEN) else 0 for node in genome.node_genes]) for genome in model.get_population()]

        best_generation_fitness = max(fitness_scores)
        fitness_variance = compute_variance(fitness_scores)
        avg_num_connections = round((sum(num_connections) / len(num_connections)), 1)
        avg_num_nodes = round((sum(num_nodes) / len(num_nodes)), 1)
        
        print(f'\nBest 5 fitnesses for generation: {sorted(fitness_scores, reverse=True)[:5]}')
        print(f'Average fitness for generation: {model.average_fitness_score}')
        print(f'Variance of fitness for generation: {fitness_variance}')
        print(f'Average number of enabled connections: {avg_num_connections}')
        print(f'Average number of nodes: {avg_num_nodes}')

        print("\nSpecies Information")
        utils.print_species_information(model.species)

        if best_generation_fitness > best_fitness:
            best_fitness = best_generation_fitness
        
            model.show_best_performer()
        model.save(f"models/best_of_generation_{generation}")



if __name__ == "__main__":
    main()