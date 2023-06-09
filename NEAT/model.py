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
from genome import Genome

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class NEAT:
	""" Class for NEAT.

        Args:
            population_size (int): initial size of the population
            env_name (string): the name of the gym environment
            weight_mutation_rate (float): probabilty of getting a connection weight mutated
			weight_random_rate (float): probability of a mutated weight being randomly assigned
			gene_disabled_rate (float): probability of an inherited gene being disabled if it was disabled in either parent
			crossover_rate (float): percentage of offspring produced by crossover
			interspecies_mating_rate (float): interspecies mating rate
			new_node_rate (float): probability of new nodes being created
			new_link__rate (float): probability of a new connection created
			dist_threshold (float): threshold used to determine if a genome belongs to a species
			coefficient1 (float): speciation coefficient 1 (excess node)
			coefficient2 (float): speciation coefficient 2  (disjoint node)
			coefficient3 (float): speciation coefficient 3  (connection weight)
    """
	def __init__(self,
		population_size=100,
		env_name="SuperMarioBros-v0",
		weight_mutation_rate=0.8,
		weight_random_rate=0.1,
		gene_disabled_rate=0.75,
		crossover_rate=0.75,
		interspecies_mating_rate=0.001,
		new_node_rate=0.03,
		new_link_rate=0.05,
		dist_threshold=3.0,
		coefficient1=1,
		coefficient2=1,
		coefficient3=1
		):

		self.population_size = population_size
		self.env_name = env_name
		self.simulation_length = 100_000

		self.weight_mutation_rate = weight_mutation_rate
		self.weight_random_rate = weight_random_rate

		self.gene_disabled_rate = gene_disabled_rate
		self.crossover_rate = crossover_rate
		self.interspecies_mating_rate = interspecies_mating_rate
		self.new_node_rate = new_node_rate
		self.new_link_rate = new_link_rate

		self.coefficient1 = coefficient1
		self.coefficient2 = coefficient2
		self.coefficient3 = coefficient3

		self.population = []
		self.species = {0: []}
		self.current_species = 1
		self.fitness_scores = []
		self.average_fitness_score = 0
		self.dist_threshold = dist_threshold

		self.input_size = (13 * 16) + 1
		self.init_connection_size = 10
		self.output_size = len(SIMPLE_MOVEMENT)

		self.env = JoypadSpace(gym_super_mario_bros.make(env_name), SIMPLE_MOVEMENT)
		self.multi_envs = MultiEnvironment(env_name, self)


	def initialise_population(self):
		""" Initialise the population with random chromosomes
        """
		for _ in range(self.population_size):
			genome = Genome(
				self.input_size, 
				self.output_size, 
				coefficient1=self.coefficient1, 
				coefficient2=self.coefficient2,
				coefficient3=self.coefficient3)
			genome.initialise_nodes()
			genome.initialise_connections(self.init_connection_size)
			self.population.append(genome)


	def initialise_species(self):
		""" Place the initial genome pool into species
		"""
		self.species[0].append(0)  # start new species

		for i in range(1, len(self.population)):  # check every genome not in a species
			genome = self.population[i]
			for k in self.species.keys():  # check every species
				first_genome = self.species[k][0]
				dist = genome.compute_distance_score(self.population[first_genome])
				if dist <= self.dist_threshold:
					self.species[k].append(i)
					break
			else:  # couldnt find a species for the genome
				self.species[self.current_species] = [i]  # create new species
				self.current_species += 1


	def evaluate_population(self):
		""" Run the simulation and get the fitness score for each individual in the population
		"""
		population, fitness_scores = self.multi_envs.run(self.population)

		self.population = list(population)
		self.fitness_scores = list(fitness_scores)

		self.average_fitness_score = sum(self.fitness_scores) // len(self.fitness_scores)


	def fitness_sharing(self):
		""" Explicit fitness sharing, genomes in larger species get their fitness scaled down more
			helps discourages species becoming too large
		"""
		for species_id, species_list in self.species.items():
			species_size = len(species_list)
			for genome_idx in species_list:
				self.fitness_scores[genome_idx] /= species_size  # fitness sharing


	def selection(self):
		pass


	def simulate_generation(self):
		self.evaluate_population()
		self.fitness_sharing()
		self.selection()
		self.crossover()
		self.speciation()




	def simulate(self, model):
		""" Simulate playing the game using a given model, used for visualisation

            Args:
                model (Genome): NEAT model for making predictions
        """
		self.env.reset()
		self.env.unwrapped.ram[1882] = 0  # set Mario's life counter to 0 to only allow one try
		action = 1
		prev_x_pos = 0
		stood_still_count = 0

		for step in range(1_000_000):
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
		        input_vector = torch.from_numpy(screen).float().flatten()
		        input_vector = torch.cat((input_vector, torch.tensor([1])))
		        
		        pred = model(input_vector)
		        pred = torch.FloatTensor(pred)

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