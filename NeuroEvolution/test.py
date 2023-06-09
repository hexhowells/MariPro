from evolution import Evolution
import selection
import crossover
import mutation
import network
import hyperparameters as hp

import torch


def main():
    world = '1'
    level = '1'

    # setup environment
    print("\nInitialising Environments...")
    evo = Evolution(
        model = network.Model,
        selection=selection.elitist_selection,
        mutation=mutation.uniform_mutation,
        env_name=f'SuperMarioBros-{world}-{level}-v0',
        population_size=1,
        gene_length=hp.gene_length,
        input_size=hp.input_size
        
        )

    model = network.Model([evo.input_length, 100, 100, evo.gene_length])
    #model = network.Model(evo.dims)  # can just use this if the model architecture is the same in Evolution
    model.load_state_dict(torch.load("model.pth"))

    evo.play_actions(model)


if __name__ == "__main__":
    main()