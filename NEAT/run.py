from genome import Genome
import numpy as np
from model import NEAT
import selection


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
		crossover_rate=0.75,
		new_node_rate=0.5,
		new_link_rate=0.25,
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
        print(f"\nGeneration: {generation} / {generations}")
        model.simulate_generation()

        fitness_scores = [round(genome.fitness, 2) for genome in model.get_population()]
        connection_sizes = [len(genome.connect_genes) for genome in model.get_population()]

        best_generation_fitness = max(fitness_scores)
        
        print(f'Best 5 fitnesses for generation: {sorted(fitness_scores, reverse=True)[:5]}')
        print(f'Average fitness for generation: {model.average_fitness_score}')
        print(f'Variance of fitness for generation: {compute_variance(fitness_scores)}')
        print(f'Average number of connections: {(sum(connection_sizes) / len(connection_sizes))}')
        
        print("Species Information")
        sorted_items = sorted(model.species.items(), key=lambda item: len(item[1]), reverse=True)
        for k, v in sorted_items:
            species_fitness_scores = [g.fitness for g in v]
            avg_fitness = round( sum(species_fitness_scores) / len(v), 1 )
            print(f'  species {k}: {len(v)}   \t{avg_fitness}\t{round(max(species_fitness_scores), 1)}')
        
        if best_generation_fitness > best_fitness:
            best_fitness = best_generation_fitness
            model.show_best_performer()



if __name__ == "__main__":
    main()