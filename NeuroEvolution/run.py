from evolution import Evolution
import selection
import crossover
import mutation
import network
import hyperparameters as hp

import wandb
import torch


def main():
    # setup environment
    print("\nInitialising Environments...")
    model = Evolution(
        population_size=hp.population_size,
        selection=selection.elitist_selection,
        crossover=None,#crossover.layer_crossover,
        mutation=mutation.random_mutation,
        env_name=hp.env_name,
        model = network.Model,
        mutation_rate=hp.mutation_rate,
        survival_rate=hp.survival_rate,
        gene_length=hp.gene_length,
        input_size=hp.input_size
        )


    model.initialise_population()

    # setup wandb (used for logging performance metrics)
    config = {
        'population_size': hp.population_size,
        'chromosome_length': model.chromosome_length,
        'gene_length': hp.gene_length,
        'survival_rate': hp.survival_rate, 
        'mutation_rate': hp.mutation_rate,
        'fitness': 'max_dist - (step/10)',
        'selection': 'elitist roulette wheel selection',
        'mutation': 'uniform mutation',
        'crossover': 'None',
        'selection_pressure': 'None',
        'model': 'MLP'
    }
    wandb.init(project='NeuroEvolution', config=config)

    # hyperparameters
    best_fitness = 0
    generations = 300

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
            torch.save(model.get_best_chromosome().state_dict(), "model.pth")

        #logs the performance metrics to wandb
        wandb.log({
            'generation':generation, 
            'best_fittness': best_fitness, 
            'average_fitness': model.average_fitness_score
            })


if __name__ == "__main__":
    main()