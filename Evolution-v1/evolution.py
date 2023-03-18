from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import time
import random
import warnings
import numpy as np

from multi_environment import MultiEnvironment

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
            crossover=None,
            env_name='SuperMarioBros-v0', 
            chromosome_length=800, 
            gene_length=7, 
            survival_rate=0.2, 
            mutation_rate=0.05,
            population_size=100
            ):

        self.initial_population_size = population_size
        self.chromosome_length = chromosome_length
        self.gene_length = gene_length
        self.survival_rate = survival_rate
        self.mutation_rate = mutation_rate
        self.simulation_length = self.chromosome_length * 5

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
            chromosome = [random.randint(0, self.gene_length-1) for _ in range(self.chromosome_length)]
            self.population.append(chromosome)


    def evaluate_population(self):
        """ Run the simulation and get the fitness score for each individual in the population
        """
        population, fitness_scores = self.multi_envs.run(self.population)

        self.population = population
        self.fitness_scores = fitness_scores

        self.average_fitness_score = sum(self.fitness_scores) // len(self.fitness_scores)


    def simulate_generation(self):
        """ Simulate an entire generation
        """
        self.evaluate_population()
        self.selection(self)
        if self.crossover: self.crossover(self)
        self.mutation(self)
        

    def play_actions(self, actions):
        """ Play a sequence of actions in the simualtor, used for visualisation

            Args:
                actions (list): list of actions
        """
        self.env.reset()
        self.env.unwrapped.ram[1882] = 0  # set Mario's life counter to 0 to only allow one try
        action = None

        for step in range(self.simulation_length):
            if step % 5 == 0: 
                action = actions[step//5]

            state, reward, done, info = self.env.step(action)

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
