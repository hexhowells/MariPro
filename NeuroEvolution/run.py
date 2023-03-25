from evolution import Evolution
import selection
import crossover
import mutation
import network

import wandb
import torch


def main():
    # setup environment
    print("\nInitialising Environments...")
    model = Evolution(
        population_size=100,
        selection=selection.elitist_selection,
        crossover=None,#crossover.one_point_crossover,
        mutation=mutation.uniform_mutation,
        env_name='SuperMarioBros-1-1-v0',
        model = network.Model
        )


    model.initialise_population()

    # setup wandb (used for logging performance metrics)
    config = {
        'population_size': 50,
        'chromosome_length': 800,
        'gene_length': 7,
        'survival_rate':0.2, 
        'mutation_rate': 0.05,
        'fitness': 'max_dist - (step/10)',
        'selection': 'elitist roulette wheel selection',
        'mutation': 'uniform mutation',
        'spread_rate': 0.1,
        'crossover': 'None',
        'selection_pressure': 'None',
        'model': 'MLP'
    }
    wandb.init(project='NeuroEvolution', config=config)

    # hyperparameters
    best_fitness = 0
    generations = 500

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