from genome import Genome
import numpy as np
from model import NEAT
import selection


def main():
    # setup environment
    print("\nInitialising Environments...")
    model = NEAT(
    	selection=selection.truncation_selection,
        population_size=200,
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

        fitness_scores = [round(genome.fitness, 2) for genome in model.get_population()]

        best_generation_fitness = max(fitness_scores)
        
        print(f'Best 5 fitnesses for generation: {sorted(fitness_scores, reverse=True)[:5]}')
        print(f'Average fitness for generation: {model.average_fitness_score}')
        
        print("Species Information")
        sorted_items = sorted(model.species.items(), key=lambda item: len(item[1]), reverse=True)
        for k, v in sorted_items:
            species_fitness_scores = [g.fitness for g in v]
            avg_fitness = round( sum(species_fitness_scores) / len(v), 1 )
            print(f'  species {k}: {len(v)}   \t{avg_fitness}\t{round(max(species_fitness_scores), 1)}')
        
        if best_generation_fitness > best_fitness:
            best_fitness = best_generation_fitness
            model.show_best_performer()

        input("")


if __name__ == "__main__":
    main()