from evolution import Evolution
import selection
import crossover
import mutation
import wandb


def save_sequence(generation, sequence):
    """ Append a sequence of actions to an output file. 
        First value is the generation the sequence came from

        Args:
            generation (int): generation the action sequence came from
            sequence (list): the sequence of actions (chromosome)
    """
    with open("sequences.txt", "a") as seq_file:
        seq_file.write(str(generation) + ",")  # store the generation along with the best sequence
        seq_file.write(','.join(map(str, sequence)))
        seq_file.write('\n\n')


def main():
    # setup environment
    print("\nInitialising Environments...")
    model = Evolution(
        population_size=100,
        selection=selection.elitist_selection,
        crossover=crossover.one_point_crossover,
        mutation=mutation.weighted_mutation,
        env_name='SuperMarioBros-1-1-v0'
        )


    model.initialise_population()

    # reset sequence history file
    with open("sequences.txt", "w") as seq_file:
        seq_file.write("")

    # setup wandb (used for logging performance metrics)
    config = {
        'population_size': 50,
        'chromosome_length': 800,
        'gene_length': 7,
        'survival_rate':0.2, 
        'mutation_rate': 0.05,
        'fitness': 'max_dist - (step/10)',
        'selection': 'elitist roulette wheel selection',
        'mutation': 'weighted mutation',
        'spread_rate': 0.1,
        'crossover': 'one-point crossover',
        'selection_pressure': 'None'
    }
    wandb.init(project='Evolution-v1', config=config)

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
            save_sequence(generation, model.get_best_chromosome())

        # logs the performance metrics to wandb
        wandb.log({
            'generation':generation, 
            'best_fittness': best_fitness, 
            'average_fitness': model.average_fitness_score
            })


if __name__ == "__main__":
    main()