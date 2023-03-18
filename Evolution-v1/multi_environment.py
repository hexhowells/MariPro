from tqdm import tqdm
from concurrent import futures

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

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
        self.batch_size = model.initial_population_size
        self.model = model
        self.envs = run_parallel(lambda idx: JoypadSpace(gym_super_mario_bros.make(name), SIMPLE_MOVEMENT), range(self.batch_size))
        self.reset()


    def reset(self):
        """ Reset all environments """
        for env in self.envs:
            env.reset()


    def get_fitness(self, env, actions):
        """ Get the fitness scores of each member in the population.
            Returns the action and fitness in order to pair the fitness scores to their respective chromosome

            Args:
                env (Evolution): evolution class instance
                actions (list): chromosome from the population
        """
        env.unwrapped.ram[1882] = 0  # set Mario's life counter to 0 to only allow one try
        action = None
        max_dist = 0

        for step in range(self.model.simulation_length):
            if step % 5 == 0: 
                action = actions[step//5]

            state, reward, done, info = env.step(action)

            if done or info['flag_get']: # cut-off simulation
                break  

            max_dist = max(max_dist, info['x_pos'])

        env.reset()
        fitness_score = int(max_dist - (step/10))  # reward travelling more distance in less steps
        return (actions, fitness_score)


    def run(self, population):
        """ Run a generation of the simulation

            Args:
                population (list): the population of chromosomes
        """
        assert len(population) == len(self.envs)
        results = run_parallel(self.get_fitness, self.envs, population)
        chromosomes, fitness_scores = zip(*results)

        return chromosomes, fitness_scores
