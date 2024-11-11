import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math


def sigmoid(x):
    """ Sigmoid activation function

    Args:
        x (float): number to pass through the activation function
    """
    return 1 / (1 + math.exp(-4.9 * x))


def draw_hitboxes(screen, hitboxes):
    """ Overlays the sprite hitboxes onto the captured game screen
    
    Args:
        screen (list): game screen (H, W, C)
        hitboxes (list): list of sprite hitboxes [(x1, y1, x2, y2), ...]
    """
    fig, ax = plt.subplots()
    ax.imshow(screen)

    for box in hitboxes:
        if sum(box) == 0 or sum(box) == 1020: continue
        x1, y1, x2, y2 = box
        h = x2 - x1
        w = y2 - y1
        rect = patches.Rectangle((x1, y1), h, w, linewidth=1, edgecolor='magenta', facecolor='magenta')
        ax.add_patch(rect)
    plt.show()


def get_center(box):
    """ Gets center point of a hitbox coordinate
    
    Args:
        box(tuple): hitbox coordinates (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box
    x = ((x2 - x1) // 2) + x1
    y = ((y2 - y1) // 2) + y1
    return x, y


def get_enemy_hitboxes(ram):
    """ Gets the enemies current hitbox coordinates
    
    Args:
        ram (list): 2KB NES ram
    """
    _enemy_hitboxes = ram[1200: 1220]
    return [_enemy_hitboxes[i-4:i] for i in range(4, len(_enemy_hitboxes), 4)]


def get_player_hitbox(ram):
    """ Gets the players current hitbox coordinates
    
    Args:
        ram (list): 2KB NES ram
    """
    return ram[1196: 1200]


def get_sprite_points(ram):
    """ Gets the center points of all player and enemy sprites (doesnt include items)

    Args:
        ram (list): 2KB NES ram
    """
    player_hitbox = get_player_hitbox(ram)
    enemy_hitboxes = get_enemy_hitboxes(ram)
    player_point = get_center(player_hitbox)
    enemy_points = [get_center(box) for box in enemy_hitboxes if 0 < sum(box) < 1020]

    return player_point, enemy_points


def _splice_buffer(buffer):
    """ Splices the (26, 16) screen buffer into a (13, 32) screen buffer
    
    Args:
        buffer (np.array): (26, 16) numpy array representing the tilemap buffer (contains tile hex strings)
    """
    buf1 = buffer[:13, :]
    buf2 = buffer[13:, :]
    
    return np.concatenate((buf1, buf2), axis=1)


def _get_screen_buffer(buffer, offset, page):
    """ Extracts the current screen tilemap (13, 16) from the tilemap buffer (13, 32)
    
    Args:
        buffer (np.array): (13, 32) numpy array representing the tilemap buffer (contains tile hex strings)
        offset (int): current x-screen offset
        page (int): current page number
    """
    
    # tilemap is stored in a 13 x 32 circular buffer
    # start of the screen is the offset + pad (not including half rendered tiles)
    pad = (page % 2) * 16  # 0 / 16

    start_col = (offset // 16) + pad
    end_col = (start_col + 16) % buffer.shape[1]

    if end_col > start_col:
        sub_arr = buffer[:, start_col:end_col]
    else:
        sub_arr = np.concatenate((buffer[:, start_col:], buffer[:, :end_col]), axis=1)

    return sub_arr


def get_screen(tiles, offset, page, display=False):
    """ Gets the current screen's tilemap from the tilemap buffer
    
    Args:
        tiles (list): the tilemap buffer (contains tile hex strings)
        offset (int): current x-screen offset
        page (int): current page number
    """
    grid = np.array(tiles).reshape((26, 16))
    if display:
        grid = np.char.ljust(grid.astype(str), width=3)
    grid = _splice_buffer(grid)
    grid = _get_screen_buffer(grid, offset, page)

    return grid


def show_screen(grid):
    """ Prints the tilemap to the console
    
    Args:
        grid (np.array): (13, 16) numpy array representing the tilemap (contains tile hex strings)
    """
    print(grid.shape)
    for row in grid:
        print('|' + ''.join(row) + '|')


def bucket_sprite(point):
    """ Places the sprites center (x, y) coordinates into the nearest tile
    
    Args:
        point (tuple): center point of the sprites hitbox
    """
    x, y = (point[0] // 16), (point[1] // 16)
    return x, min(12, (y-2))


def get_input_screen(ram):
    """ Gets the tilemap of the screen with player and enemy tiles overlapped
    
    Args:
        ram (list): 2Kb NES RAM
    """

    # get tilemap grid
    page = ram[1818]
    offset = ram[1820]
    tiles = [hex(t)[2:] for t in ram[1280:1696]]
    grid = get_screen(tiles, offset, page)#.astype(np.float)
    grid = np.where(grid != '0', 1, 0)

    player, enemies = get_sprite_points(ram)
    p_col, p_row = bucket_sprite(player)
    grid[p_row][p_col] = 2

    for enemy in enemies:
        e_col, e_row = bucket_sprite(enemy)
        grid[e_row][e_col] = -1

    return grid


def print_species_information(species):
    sorted_items = sorted(species.items(), key=lambda item: len(item[1]), reverse=True)

    print('\t\tpopulation size\t   average fitness\t max fitness')
    for k, v in sorted_items:
        species_fitness_scores = [g.fitness for g in v]
        avg_fitness = round( sum(species_fitness_scores) / len(v), 1 )
        if len(v) > 1:
            print(f'  species {k}: \t{len(v)}\t\t   {avg_fitness}\t\t\t {round(max(species_fitness_scores), 1)}')
