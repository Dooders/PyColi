import argparse
import io
import random

import imageio
import matplotlib.pyplot as plt
import numpy as np


def initialize_grids(grid_size_x: int, grid_size_y: int) -> tuple:
    """
    Initialize nutrient and repellent grids with radial gradients.

    Parameters
    ----------
    grid_size_x : int
        The horizontal size of the grid.
    grid_size_y : int
        The vertical size of the grid.

    Returns
    -------
    tuple
        Tuple containing the nutrient grid and repellent grid.
        Each grid is a 2D numpy array.
    """
    nutrient_grid = np.zeros((grid_size_x, grid_size_y))
    repellent_grid = np.zeros((grid_size_x, grid_size_y))
    center_x, center_y = grid_size_x // 2, grid_size_y // 2

    # Populate grids with radial gradients
    for x in range(grid_size_x):
        for y in range(grid_size_y):
            distance_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            nutrient_grid[x, y] = np.exp(-distance_to_center / 20.0)
            repellent_grid[x, y] = 1 - np.exp(-distance_to_center / 20.0)

    return nutrient_grid, repellent_grid


def initialize_bacteria(grid_size_x: int, grid_size_y: int, option='center') -> list:
    """
    Initialize the starting position of bacteria.

    Parameters
    ----------
    grid_size_x : int
        The horizontal size of the grid.
    grid_size_y : int
        The vertical size of the grid.
    option : str, optional
        The initialization option. The default is 'center'.

    Returns
    -------
    list
        List containing the starting position of bacteria.
    """
    if option == 'center':
        return [(grid_size_x // 2, grid_size_y // 2)]
    elif option == 'random':
        x = random.randint(0, grid_size_x - 1)
        y = random.randint(0, grid_size_y - 1)
        return [(x, y)]
    elif option == 'top_left':
        x = int(grid_size_x * .25)
        y = int(grid_size_y * .25)
        return [(x, y)]
    elif option == 'bottom_right':
        return [(grid_size_x - 1, 0)]
    else:
        raise ValueError("Invalid option. Choose 'center' or 'random'.")


def get_level(x: int, y: int, grid: np.ndarray) -> float:
    """
    Get the nutrient or repellent level at a specific position in a grid.

    Parameters
    ----------
    x : int
        The horizontal position in the grid.
    y : int
        The vertical position in the grid.
    grid : np.ndarray
        The grid to get the level from.

    Returns
    -------
    float
        The nutrient or repellent level at the given position.
    """
    if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
        return grid[x, y]
    else:
        return 0


def get_possible_moves(x: int, y: int) -> list:
    """
    Get possible moves from current position.

    Parameters
    ----------
    x : int
        The horizontal position in the grid.
    y : int
        The vertical position in the grid.

    Returns
    -------
    list
        List containing the possible moves.
    """
    # Define possible moves: up, down, left, right
    possible_moves = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

    return possible_moves


def random_walk(x: int, y: int, *args, **kwargs) -> tuple:
    """ 
    Randomly select a move from the possible moves.

    Parameters
    ----------
    x : int
        The horizontal position in the grid.
    y : int
        The vertical position in the grid.

    Returns
    -------
    tuple
        Tuple containing the new position.
    """
    possible_moves = get_possible_moves(x, y)

    return possible_moves[np.random.choice(range(4))]


def biased_random_walk(x: int,
                       y: int,
                       nutrient_grid: np.ndarray,
                       repellent_grid: np.ndarray) -> tuple:
    """
    Evaluate possible moves based on current position and grids and select a 
    move based on the bias of nutrient and repellent levels.

    Parameters
    ----------
    x : int
        The horizontal position in the grid.
    y : int
        The vertical position in the grid.
    nutrient_grid : np.ndarray
        The nutrient grid.
    repellent_grid : np.ndarray
        The repellent grid.

    Returns
    -------
    tuple
        Tuple containing the possible moves and their respective biases.
    """
    # Define possible moves: up, down, left, right
    possible_moves = get_possible_moves(x, y)

    # Evaluate nutrient and repellent levels at possible moves
    possible_nutrient_levels = [
        get_level(x, y, nutrient_grid) for x, y in possible_moves]
    possible_repellent_levels = [
        get_level(x, y, repellent_grid) for x, y in possible_moves]

    # Calculate biases for each possible move
    bias = np.exp(possible_nutrient_levels) / \
        (1 + np.exp(possible_repellent_levels))
    bias /= np.sum(bias)

    new_x, new_y = possible_moves[np.random.choice(range(4), p=bias)]

    return new_x, new_y


WALK_FUNCTIONS = {
    'random_walk': random_walk,
    'biased_random_walk': biased_random_walk,
}


def get_move(walk_function: str,
             x: int,
             y: int,
             nutrient_grid: np.ndarray,
             repellent_grid: np.ndarray) -> tuple:
    """ 
    Get the next move based on the walk function.

    Parameters
    ----------
    walk_function : str
        The walk function to use.
    x : int
        The horizontal position in the grid.
    y : int
        The vertical position in the grid.
    nutrient_grid : np.ndarray
        The nutrient grid.
    repellent_grid : np.ndarray
        The repellent grid.

    Returns
    -------
    tuple
        Tuple containing the new position.
    """

    return WALK_FUNCTIONS[walk_function](x, y, nutrient_grid, repellent_grid)


def visualize_state(grid: np.ndarray, bacteria_positions: list, time_step: int) -> None:
    """
    Visualize the current state of the simulation.

    Parameters
    ----------
    grid : np.ndarray
        The grid to visualize.
    bacteria_positions : list
        List containing the bacteria positions.
    time_step : int
        The current time step.
    """
    plt.imshow(grid, cmap='viridis')

    bacteria_x, bacteria_y = zip(*bacteria_positions)

    plt.scatter(bacteria_y, bacteria_x, c='red', label='Bacteria')
    plt.colorbar(label='Nutrient Concentration')
    plt.title(f'Time step {time_step}')

    # Remove tick labels
    plt.xticks([])
    plt.yticks([])

    plt.pause(0.1)

    # Save image to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Read image from buffer
    image_array = imageio.v2.imread(buf)

    # Clear buffer and plot
    plt.clf()

    return image_array


class Bacteria:

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.previous_concentration = 0
        
    def move(self, simulation):

        nearby_positions = self.nearby

        # Evaluate nutrient and repellent levels at possible moves
        nutrient_levels = [
            get_level(x, y, simulation.nutrient_grid) for x, y in nearby_positions]
        repellent_levels = [
            get_level(x, y, simulation.repellent_grid) for x, y in nearby_positions]
        
        if simulation.move_function == 'biased_random_walk':
            bias = np.exp(nutrient_levels) / (1 + np.exp(repellent_levels))
            
        elif simulation.move_function == 'biased_random_walk_with_memory':
            # Adjust bias for cells with nutrient levels higher than previous level
            memory_factor = np.array([2 if level > self.previous_concentration else 1 for level in nutrient_levels])
            bias = np.exp(nutrient_levels) * memory_factor / (1 + np.exp(repellent_levels))
            
        else:
            bias = np.ones(len(nearby_positions)) / len(nearby_positions)

        bias /= np.sum(bias)
        new_x, new_y = nearby_positions[np.random.choice(range(4), p=bias)]
        
        # Apply boundary conditions
        new_x = min(max(0, new_x), simulation.grid_size_x - 1)
        new_y = min(max(0, new_y), simulation.grid_size_y - 1)
        
        self.position = (new_x, new_y)
        
        self.previous_concentration = get_level(new_x, new_y, simulation.nutrient_grid)

        return new_x, new_y

    @property
    def nearby(self):
        return [(self.x-1, self.y), (self.x+1, self.y),
                (self.x, self.y-1), (self.x, self.y+1)]
        
    @property
    def position(self):
        return (self.x, self.y)
    
    @position.setter
    def position(self, new_position):
        self.x, self.y = new_position


class Simulation:

    def __init__(self, time_steps: int,
                 grid_size_x: int,
                 grid_size_y: int,
                 consumption_rate: float,
                 move_function: str,
                 placement: str):

        self.time_steps = time_steps
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.consumption_rate = consumption_rate
        self.move_function = move_function
        self.placement = placement

        self.bacteria_positions = initialize_bacteria(
            self.grid_size_x, self.grid_size_y, self.placement)

        self.nutrient_grid, self.repellent_grid = initialize_grids(
            self.grid_size_x, self.grid_size_y)
        
        self._setup()
        
    def _setup(self):
        self.bacteria = [Bacteria(x, y) for x, y in self.bacteria_positions]

    def _simulate(self, t: int):
        new_positions = []
        for bacteria in self.bacteria:
            # Get next move
            new_x, new_y = bacteria.move(self)

            new_positions.append((new_x, new_y))

        # Update bacteria positions for the next iteration
        self.bacteria_positions = new_positions

        # Visualize the simulation state every 10 time steps
        if t % 1 == 0:
            image = visualize_state(
                self.nutrient_grid, self.bacteria_positions, t)

            self.image_list.append(image)

    def simulate(self) -> None:
        """ 
        Simulate the movement of bacteria in a grid.

        Parameters
        ----------
        grid_size_x : int
            The horizontal size of the grid.
        grid_size_y : int
            The vertical size of the grid.
        consumption_rate : float
            The rate at which bacteria consume nutrient.
        move_function : str
            The move function to use
        placement : str
            The placement option for bacteria. The default is 'center'.
        """

        self.image_list = []

        # Main simulation loop
        for t in range(self.time_steps):
            self._simulate(t)

        # Save the results in a GIF
        imageio.mimsave('bacteria.gif', self.image_list, fps=20)

    def get_levels(self, bacteria: Bacteria, level_name: str):
        return [get_level(x, y, getattr(self, f'{level_name}_grid')) for x, y in bacteria.nearby]


if __name__ == '__main__':

    # command line update defaults
    parser = argparse.ArgumentParser(description='Bacteria simulation')
    parser.add_argument('--time_steps', type=int,
                        default=100, help='number of time steps')
    parser.add_argument('--grid_size_x', type=int,
                        default=25, help='grid size x')
    parser.add_argument('--grid_size_y', type=int,
                        default=25, help='grid size y')
    parser.add_argument('--consumption_rate', type=float,
                        default=.1, help='consumption rate')
    parser.add_argument('--move_function', type=str,
                        default='biased_random_walk_with_memory', help='move function')
    parser.add_argument('--placement', type=str,
                        default='top_left', help='placement option')

    args = parser.parse_args()

    simulation = Simulation(args.time_steps, args.grid_size_x, args.grid_size_y,
                            args.consumption_rate, args.move_function, args.placement)

    simulation.simulate()
