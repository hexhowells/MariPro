from tqdm import tqdm
from concurrent import futures

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import torch
import utils

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def run_parallel(fn, *argv):
    """ Run a function in a thread pool

        Args:
            fn (function): the function to execute
            argv (parameters): the parameters to pass to the function
    """
    with futures.ThreadPoolExecutor(max_workers=12) as executor:
        result_iterator = list(tqdm(executor.map(fn, *argv)))
    return [i for i in result_iterator]


class MultiEnvironment():
    """ Class allows multiple gym environments to be run asynchronously

        Args:
            name (string): name of the mario gym environment
            model (Evolution): evolution model class
    """
    def __init__(self, name, model):
        self.batch_size = model.population_size
        self.model = model
        self.envs = run_parallel(lambda idx: JoypadSpace(gym_super_mario_bros.make(name), SIMPLE_MOVEMENT), range(self.batch_size))
        self.reset()


    def __del__(self):
        for env in self.envs:
            env.close()


    def reset(self):
        """ Reset all environments """
        for env in self.envs:
            env.reset()


    def get_fitness(self, env, model):
        """ Get the fitness scores of each member in the population.
            Returns the action and fitness in order to pair the fitness scores to their respective chromosome

            Args:
                env (Evolution): evolution class instance
                actions (list): chromosome from the population
        """
        env.unwrapped.ram[1882] = 0  # set Mario's life counter to 0 to only allow one try
        action = 1
        max_dist = 0
        prev_x_pos = 0
        stood_still_count = 0

        for step in range(self.model.simulation_length):
            state, reward, done, info = env.step(action)

            if step % 5 == 0: 
                player_x_pos = info['x_pos']

                if player_x_pos <= prev_x_pos:
                    stood_still_count += 1
                else:
                    stood_still_count = 0

                if stood_still_count > 20:
                    break

                prev_x_pos = player_x_pos
                
                screen = utils.get_input_screen(env.unwrapped.ram)
                input_vector = torch.from_numpy(screen).float().flatten()
                input_vector = torch.cat((input_vector, torch.tensor([1])))
                
                pred = model(input_vector.numpy())
                pred = torch.FloatTensor(pred)

                action = torch.argmax(pred).item()

            if done or info['flag_get']: # cut-off simulation
                break  

            max_dist = max(max_dist, info['x_pos'])

        env.reset()
        fitness_score = int(max_dist - (step/10))  # reward travelling more distance in less steps
        fitness_score = max(1, fitness_score)
        model.fitness = fitness_score
        


    def run(self, population):
        """ Run a generation of the simulation

            Args:
                population (list): the population of chromosomes
        """
        assert len(population) == len(self.envs), f'population size [{len(population)}] does not match environment size [{len(self.envs)}]'
        run_parallel(self.get_fitness, self.envs, population)

