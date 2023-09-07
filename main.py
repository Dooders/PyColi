import random

import matplotlib.pyplot as plt
import numpy as np

# Initialize grid dimensions
grid_size_x, grid_size_y = 100, 100

# Initialize a list of bacteria, starting with one at the center
bacteria_positions = [(grid_size_x // 2, grid_size_y // 2)]

# Initialize nutrient and repellent grids with radial gradients
nutrient_grid = np.zeros((grid_size_x, grid_size_y))
repellent_grid = np.zeros((grid_size_x, grid_size_y))
center_x, center_y = grid_size_x // 2, grid_size_y // 2

# Populate grids
for x in range(grid_size_x):
    for y in range(grid_size_y):
        distance_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        nutrient_grid[x, y] = np.exp(-distance_to_center / 20.0)
        repellent_grid[x, y] = 1 - np.exp(-distance_to_center / 40.0)

# Define consumption rate and time step
consumption_rate = 0.1
time_step = 1

# Define function to get nutrient or repellent level at a position
def get_level(x, y, grid):
    if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
        return grid[x, y]
    else:
        return 0

# Simulation loop
for t in range(1000):
    new_positions = []
    for x, y in bacteria_positions:
        # Update nutrient grid based on bacteria position
        current_nutrient_level = get_level(x, y, nutrient_grid)
        nutrient_consumed = min(current_nutrient_level, consumption_rate)
        nutrient_grid[x, y] -= nutrient_consumed

        # Possible moves: up, down, left, right
        possible_moves = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

        # Evaluate nutrient and repellent levels at possible moves
        possible_nutrient_levels = [get_level(x, y, nutrient_grid) for x, y in possible_moves]
        possible_repellent_levels = [get_level(x, y, repellent_grid) for x, y in possible_moves]

        # Calculate bias for random choice
        bias = np.exp(possible_nutrient_levels) / (1 + np.exp(possible_repellent_levels))
        bias /= np.sum(bias)  # Normalize to form a probability distribution

        # Choose next position based on bias
        new_x, new_y = possible_moves[np.random.choice(range(4), p=bias)]

        # Boundary conditions
        new_x = min(max(0, new_x), grid_size_x - 1)
        new_y = min(max(0, new_y), grid_size_y - 1)
        new_positions.append((new_x, new_y))

    # Update bacteria positions
    bacteria_positions = new_positions

    # Visualization (every 10 time steps)
    if t % 100 == 0:
        plt.imshow(nutrient_grid, cmap='viridis')
        bacteria_x, bacteria_y = zip(*bacteria_positions)
        plt.scatter(bacteria_y, bacteria_x, c='red', label='Bacteria')
        plt.colorbar(label='Nutrient Concentration')
        plt.title(f'Time step {t}')
        plt.pause(0.1)
        plt.clf()
        clear_output()
        

# Show the final state
plt.imshow(nutrient_grid, cmap='viridis')
bacteria_x, bacteria_y = zip(*bacteria_positions)
plt.scatter(bacteria_y, bacteria_x, c='red', label='Bacteria')
plt.colorbar(label='Nutrient Concentration')
plt.title('Final State')
plt.show()

