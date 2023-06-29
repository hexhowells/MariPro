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
from crossover import crossover as crossover_fn

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class NEAT:
	""" Class for NEAT.

        Args:
        	selection (function): the selection function to apply
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
			culling_factor (float): value for determining diversity in species
			coefficient1 (float): speciation coefficient 1 (excess node)
			coefficient2 (float): speciation coefficient 2  (disjoint node)
			coefficient3 (float): speciation coefficient 3  (connection weight)
    """
	def __init__(self,
		selection,
		population_size=100,
		env_name="SuperMarioBros-v0",
		survival_rate=0.2,
		weight_mutation_rate=0.8,
		weight_random_rate=0.1,
		gene_disabled_rate=0.75,
		crossover_rate=0.75,
		interspecies_mating_rate=0.001,
		new_node_rate=0.03,
		new_link_rate=0.05,
		weight_disable_rate=0.4,
		weight_enable_rate=0.2,
		dist_threshold=0.3,
		culling_factor=1,
		coefficient1=1,
		coefficient2=1,
		coefficient3=1
		):

		self.population_size = population_size
		self.env_name = env_name
		self.simulation_length = 100_000
		self.selection_fn = selection

		self.survival_rate = survival_rate
		self.weight_mutation_rate = weight_mutation_rate
		self.weight_random_rate = weight_random_rate

		self.gene_disabled_rate = gene_disabled_rate
		self.crossover_rate = crossover_rate
		self.interspecies_mating_rate = interspecies_mating_rate
		self.new_node_rate = new_node_rate
		self.new_link_rate = new_link_rate

		self.weight_disable_rate = weight_disable_rate
		self.weight_enable_rate = weight_enable_rate

		self.coefficient1 = coefficient1
		self.coefficient2 = coefficient2
		self.coefficient3 = coefficient3

		self.population = []
		self.species = {0: []}
		self.current_species = 1
		self.average_fitness_score = 0
		self.dist_threshold = dist_threshold
		self.culling_factor = culling_factor

		self.input_size = (13 * 16) + 1
		self.init_connection_size = 8
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
			
			#[genome.mutate_node() for _ in range(14)]

			self.population.append(genome)


	def initialise_species(self):
		""" Place the initial genome pool into species
		"""
		self.species[0].append(self.population[0])  # start new species

		for i in range(1, len(self.population)):  # check every genome not in a species
			genome = self.population[i]
			for k in self.species.keys():  # check every species
				first_genome = self.species[k][0]
				dist = genome.compute_distance_score(first_genome)
				if dist <= self.dist_threshold:
					self.species[k].append(self.population[i])
					break
			else:  # couldnt find a species for the genome
				self.species[self.current_species] = [self.population[i]]  # create new species
				self.current_species += 1

		self.population = []


	def get_population(self):
		for species in self.species.values():
			for genome in species:
				yield genome


	def get_population_with_species(self):
		for species_id, species in self.species.items():
			for genome in species:
				yield (genome, species_id)


	def get_population_size(self):
		return len(list(self.get_population()))


	def evaluate_population(self):
		""" Run the simulation and get the fitness score for each individual in the population
		"""
		self.multi_envs.run(list(self.get_population()))

		fitness_scores = [genome.fitness for genome in self.get_population()]

		self.average_fitness_score = sum(fitness_scores) // len(fitness_scores)


	def fitness_sharing(self):
		""" Explicit fitness sharing, genomes in larger species get their fitness scaled down more
			helps discourages species becoming too large
		"""
		for species in self.species.values():
			species_size = len(species)
			for genome in species:
				genome.fitness = genome.fitness / (species_size * 0.25)  # fitness sharing


	def get_average_species_fitness(self):
		""" Get average fitness scores for each species
		"""
		avg_species_fitness_scores = []

		for species_id, species in self.species.items():
			_fitness_scores = [genome.fitness for genome in species]
			avg_fitness = sum(_fitness_scores) / len(_fitness_scores)
			avg_species_fitness_scores.append((species_id, avg_fitness))

		return avg_species_fitness_scores


	def get_total_adjusted_fitness(self, avg_fitness_scores):
		""" Get total adjusted fitness scores for each species

			Args:
				avg_fitness_scores (list): (species id, average fitness scores for each species)
		"""
		adj_fitness_scores = []

		for (species_id, avg_fitness) in avg_fitness_scores:
			adj_fitness = avg_fitness / len(self.species[species_id]) * self.culling_factor
			adj_fitness_scores.append((species_id, avg_fitness))

		return adj_fitness_scores


	def get_offspring_rates(self, adj_fitness_scores):
		""" Get offspring rates for each species

			Args:
				adj_fitness_scores (list): (species id, adjusted fitness scores for each species)
		"""
		total_adj_fitness = sum([score[1] for score in adj_fitness_scores])
		num_offspring = {}
		num_new_offspring = self.population_size - self.get_population_size()

		for (species_id, adj_fitness) in adj_fitness_scores:
			offspring = num_new_offspring * (adj_fitness / total_adj_fitness)
			num_offspring[species_id] = int(offspring)

		species_ids = list(num_offspring.keys())
		remaining_offspring = num_new_offspring - sum(num_offspring.values())
		for _ in range(remaining_offspring):
			num_offspring[random.choice(species_ids)] += 1

		return num_offspring


	def selection(self):
		""" Apply a selection function to select survivors for the next generation
		"""
		population_with_species_id = list(self.get_population_with_species())
		species_fitness = [genome.fitness for (genome, _) in population_with_species_id]
		selected_genomes = self.selection_fn(population_with_species_id, species_fitness, self.survival_rate)
		
		self.species = {}
		for genome, species_id in selected_genomes:
			if species_id in self.species:
				self.species[species_id].append(genome)
			else:
				self.species[species_id] = [genome]


	def cull_species(self):
		""" Remove any species with only a single member since they cannot breed
		"""
		species_to_cull = []
		for species_id, species in self.species.items():
			if len(species) < 2:
				species_to_cull.append(species_id)

		for species_id in species_to_cull:
			self.species.pop(species_id, None)


	def crossover(self, offspring_rates):
		offspring = []
		for species_id, species in self.species.items():
			fitness_scores = [genome.fitness for genome in species]
			probs = np.asarray(fitness_scores) / sum(fitness_scores)

			for _ in range(offspring_rates[species_id]):
				parent1, parent2 = np.random.choice(species, size=2, replace=False, p=probs)
				child = crossover_fn(parent1, parent2)
				child.fitness = (parent1.fitness + parent2.fitness) / 2
				offspring.append(child)
				
		return offspring


	def speciation(self, offspring):
		""" Assign each new offspring to a species

			Args:
				offspring (list): list of offspring Genomes to be speciated
		"""
		num_parents = self.get_population_size()
		for genome in offspring:  # check each genome in the new offspring

			for k in self.species.keys():  # check every species
				first_genome = self.species[k][0]
				dist = genome.compute_distance_score(first_genome)
				if dist <= self.dist_threshold:
					self.species[k].append(genome)
					break
			else:  # couldnt find a species for the genome
				self.species[self.current_species] = [genome]  # create new species
				self.current_species += 1


	def structural_mutation(self, population):
		""" Mutate node or connection genes
		"""
		for genome in population:
			if random.random() < self.new_node_rate:
				genome.mutate_node()

			if random.random() < self.new_link_rate:
				genome.mutate_connection()

		return population


	def weight_mutation(self, population):
		""" Mutate connection weights in genome
		"""
		for genome in population:
			genome.mutate_weight(
				self.weight_mutation_rate, 
				self.weight_random_rate, 
				self.weight_disable_rate,
				self.weight_enable_rate)

		return population


	def simulate_generation(self):
		self.evaluate_population()
		#self.fitness_sharing()

		self.selection()
		self.cull_species()

		avg_species_fitness = self.get_average_species_fitness()
		adj_species_fitness = self.get_total_adjusted_fitness(avg_species_fitness)
		offspring_rates = self.get_offspring_rates(adj_species_fitness)
		[print(f'> species {species_id} rate: {rate}') for species_id, rate in offspring_rates.items()]

		new_offspring = self.crossover(offspring_rates)
		new_offspring = self.structural_mutation(new_offspring)
		new_offspring = self.weight_mutation(new_offspring)

		self.speciation(new_offspring)

		self.evaluate_population()


	def simulate(self, model):
		""" Simulate playing the game using a given model, used for visualisation

            Args:
                model (Genome): NEAT model for making predictions
        """
		self.env.reset()
		self.env.unwrapped.ram[1882] = 0  # set Mario's life counter to 0 to only allow one try
		action = 1
		max_dist = 0
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

		    max_dist = max(max_dist, info['x_pos'])

		self.env.viewer.close()
		fitness_score = int(max_dist - (step/10))  # reward travelling more distance in less steps
		fitness_score = max(0, fitness_score)
		print("fitness score: ", fitness_score)


	def get_best_chromosome(self):
		""" Get the current best chromosome from the population
		"""
		best_fitness = -1
		best_genome = None
		for species in self.species.values():
			for genome in species:
				if (fitness_score := genome.fitness) > best_fitness:
					best_genome = genome
					best_fitness = fitness_score

		return best_genome


	def show_best_performer(self):
		""" Play the actions of the best performing individual in the population
		"""
		best_chromosome = self.get_best_chromosome()
		self.simulate(best_chromosome)