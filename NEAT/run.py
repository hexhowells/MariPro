from genome import Genome
import numpy as np
from model import NEAT
import selection


def main():
    # setup environment
    print("\nInitialising Environments...")
    model = NEAT(
    	selection=selection.truncation_selection,
        population_size=100,
		env_name="SuperMarioBros-v0",
		weight_mutation_rate=0.8,
		weight_random_rate=0.1,
		gene_disabled_rate=0.75,
		crossover_rate=0.75,
		interspecies_mating_rate=0.001,
		new_node_rate=0.03,
		new_link_rate=0.05,
		coefficient1=1,
		coefficient2=1,
		coefficient3=1
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

        best_generation_fitness = max(model.fitness_scores)

        print(f'Best 5 fitnesses for generation: {sorted(model.fitness_scores, reverse=True)[:5]}')
        print(f'Average fitness for generation: {model.average_fitness_score}')
        
        if best_generation_fitness > best_fitness:
            best_fitness = best_generation_fitness
            model.show_best_performer()


if __name__ == "__main__":
    main()