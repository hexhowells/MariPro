from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import time
import warnings

import utils

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def find_pits(grid):
    """ Find the columns that have pits/holes (possible for Mario to fall into)
    
    Args:
        grid (np.array): (13, 16) numpy array representing the tilemap (contains tile hex strings)
    """
    pits = []
    for i, col in enumerate(grid.T):
        if col[-1] == '0':
            pits.append(i)

    return pits


def get_floor_heights(col):
    """ Get the lowest tile that contains a platform for each column in the tilemap
    
    Args:
        col (np.array): (13, 1) numpy array representing a column of the tilemap (contains tile hex strings)
    """
    for i, tile in enumerate(reversed(col)):
        if tile == '0':
            return i
    return i


def detect_obstacle(grid, by):
    """ Detect a platform (non floating) that is above Mario's current height
    
    Args:
        grid (np.array): (13, 16) numpy array representing the tilemap (contains tile hex strings)
        by (int): the y coordinate of the tilemap that Mario currently occupies (the row in the tilemap grid)
    """
    by = 13 - by
    obstacles = []
    for i, col in enumerate(grid.T):
        h = get_floor_heights(col)
        if h >= by:
            obstacles.append((i, h))

    return obstacles


def get_action(ram):
    """ Select an action from the current game state using a set of heuristics

        The action selected represents the jump duration. return of 0 means mario doesnt jump and will
        default to running to the right. A jump duration >0 will press the jump button for n frames
    
    Args:
        ram (list): 2KB NES ram
    """

    # get player and enemy coordinates
    player, enemies = utils.get_sprite_points(ram)
    bx, by = utils.bucket_sprite(player)

    # get tilemap grid
    page = ram[1818]
    offset = ram[1820]
    tiles = [hex(t)[2:] for t in ram[1280:1696]]
    grid = utils.get_screen(tiles, offset, page)

    #
    # Choose action
    #

    # jump over pits
    x_pits = find_pits(grid)

    if len(x_pits) > 0:
        dist = x_pits[0] - bx
        if -1 <= dist < 2:
            return 20

    # jump over enemies
    for enemy in enemies:
        xdiff = enemy[0] - player[0]
        ydiff = enemy[1] - player[1]
        
        if ydiff > 30 and (0 < xdiff < 70):  # jump earlier if enemy is below mario
            return 15
        elif ydiff >= -4 and 0 < xdiff < 25:  # jump over enemy
            return 6

    # jump over obstacles
    obstacles = detect_obstacle(grid, by)

    if len(obstacles) > 0:
        for ob in obstacles:
            dist = ob[0] - bx
            if 0 < dist < 4:
                return 20 

    # dont jump (default to running)
    return 0


# setup environment
_env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(_env, SIMPLE_MOVEMENT)

# get the NES RAM
ram = env.unwrapped.ram

# define action indexes
ACTION_RIGHT = 1
ACTION_RIGHT_JUMP = 2

done = True
action = ACTION_RIGHT
jump_counter = 0  # how many frames to hold jump for

# run simulation
for step in range(1_000_000):
    if done: state = env.reset()

    state, reward, done, info = env.step(action)
    env.render()

    # perform action
    if jump_counter > 0:
        jump_counter -= ACTION_RIGHT
        action = ACTION_RIGHT_JUMP
    else:
        action = ACTION_RIGHT
        jump_counter = get_action(ram)

    time.sleep(0.01)

env.close()