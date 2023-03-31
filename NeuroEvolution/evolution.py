from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import time
import random
import warnings
import numpy as np
import torch
import copy

from multi_environment import MultiEnvironment
import utils

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Evolution:
    """ Class for evolutionary learning.

        Args:
            selection (function): the selection function to apply
            mutation (function): the mutation function to apply
            crossover (function): the crossover function to apply
            env_name (string): the name of the gym environment
            chromosome_length (int): length of the chromosome
            gene_length (int): length of the gene
            survival_rate (float): percentage of the population to keep during selection
            mutation_rate (float): percentage of genes to mutate
            population_size (int): initial size of the population
    """

    def __init__(self,
            selection,
            mutation,
            model,
            gene_length,
            input_size,
            crossover=None,
            env_name='SuperMarioBros-v0',
            survival_rate=0.2, 
            mutation_rate=0.05,
            population_size=100
            ):

        self.initial_population_size = population_size
        self.input_length = input_size[0] * input_size[1]
        self.gene_length = gene_length
        self.chromosome_length = self.input_length * self.gene_length
        
        self.survival_rate = survival_rate
        self.mutation_rate = mutation_rate
        self.simulation_length = self.chromosome_length * 5

        self.model = model
        self.dims = [self.input_length, 100, 100, self.gene_length]

        self.env = JoypadSpace(gym_super_mario_bros.make(env_name), SIMPLE_MOVEMENT)
        self.multi_envs = MultiEnvironment(env_name, self)
        
        self.population = []
        self.fitness_scores = []
        self.average_fitness_score = 0

        self.selection = selection
        self.mutation = mutation
        self.crossover = crossover


    def initialise_population(self):
        """ Initialise the population with random chromosomes
        """
        for _ in range(self.initial_population_size):
            chromosome = self.model(self.dims)
            self.population.append(chromosome)


    def evaluate_population(self):
        """ Run the simulation and get the fitness score for each individual in the population
        """
        population, fitness_scores = self.multi_envs.run(self.population)

        self.population = list(population)
        self.fitness_scores = list(fitness_scores)

        self.average_fitness_score = sum(self.fitness_scores) // len(self.fitness_scores)


    def select_offspring(self):
        """ Select offspring from the surviving parents
            Offspring can then be modified using crossover or mutation
        """
        probs = np.asarray(self.fitness_scores) / sum(self.fitness_scores)

        # determine how many chromosomes to create
        original_pop_size = len(self.population) / self.survival_rate
        size = int( original_pop_size - len(self.population) )

        offspring_indexes = np.random.choice(range(len(self.population)), size=size, p=probs)

        offspring, fitness = [], []

        for idx in offspring_indexes:
            offspring.append(copy.deepcopy(self.population[idx]))
            fitness.append(self.fitness_scores[idx])

        return offspring, fitness


    def simulate_generation(self):
        """ Simulate an entire generation
        """
        self.evaluate_population()
        self.selection(self)

        if self.crossover: 
            offspring, fitness = self.crossover(self)
        else:
            offspring, fitness = self.select_offspring()

        self.mutation(self, offspring, fitness)
        

    def play_actions(self, model):
        """ Play a sequence of actions in the simualtor, used for visualisation

            Args:
                actions (list): list of actions
        """
        self.env.reset()
        self.env.unwrapped.ram[1882] = 0  # set Mario's life counter to 0 to only allow one try
        action = 1
        prev_x_pos = 0
        stood_still_count = 0

        for step in range(self.simulation_length):
            state, reward, done, info = self.env.step(action)

            if step % 5 == 0: 
                player_x_pos = info['x_pos']

                if player_x_pos <= prev_x_pos:
                    stood_still_count += 1
                else:
                    stood_still_count = 0

                if stood_still_count > 20:
                    break

                prev_x_pos = player_x_pos
                
                screen = utils.get_input_screen(self.env.unwrapped.ram)
                pred = model(torch.from_numpy(screen).float())
                action = torch.argmax(pred).item()

            self.env.render()
            time.sleep(0.01)

            if done or info['flag_get']: # cut-off simulation
                break  

        self.env.viewer.close()

    
    def get_best_chromosome(self):
        """ Get the current best chromosome from the population
        """
        best_idx = self.fitness_scores.index(max(self.fitness_scores))
        return self.population[best_idx]


    def show_best_performer(self):
        """ Play the actions of the best performing individual in the population
        """
        best_chromosome = self.get_best_chromosome()
        self.play_actions(best_chromosome)

def hash_model_parameters(model):
    """
    Creates a hash of the parameters of a SimpleModel instance.

    Parameters:
        model (SimpleModel): The model to hash.

    Returns:
        str: A string hash of the model parameters.
    """
    # Get the parameters of the model as numpy arrays.
    param_arrays = [param.data.numpy() for param in model.parameters()]

    # Concatenate the parameter arrays into a single numpy array.
    flat_params = np.concatenate([arr.flatten() for arr in param_arrays])

    # Compute the hash of the flattened parameter array.
    hash_value = hash(flat_params.tobytes())

    return str(hash_value)