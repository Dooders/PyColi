import numpy as np

import random
import math

class EColi:
    def __init__(self, position, sensing_range, medium_size):
        self.position = np.array(position, dtype=np.float64)
        self.sensing_range = sensing_range
        self.medium_size = medium_size
        self.direction_angle = random.uniform(0, 2 * math.pi)  # Initial direction angle in radians
        self.run_duration = random.randint(5, 20)  # Duration of a run
        self.steps_since_tumble = 0
        self.previous_nutrient = 0
        self.previous_toxin = 0
        self.methylation_level = 0.5  # Memory to adjust chemotactic sensitivity (value between 0 and 1)

    def sense(self, field):
        x, y = self.position
        x_min = max(0, int(x - self.sensing_range))
        x_max = min(self.medium_size[0], int(x + self.sensing_range))
        y_min = max(0, int(y - self.sensing_range))
        y_max = min(self.medium_size[1], int(y + self.sensing_range))

        field_concentration = np.sum(field[x_min:x_max, y_min:y_max])
        return field_concentration

    def run_and_tumble(self, nutrient_field, toxin_field):
        current_nutrient = self.sense(nutrient_field)
        current_toxin = self.sense(toxin_field)
        nutrient_gradient = current_nutrient - self.previous_nutrient
        toxin_gradient = current_toxin - self.previous_toxin

        # Update methylation level to regulate responsiveness
        if nutrient_gradient > 0:
            self.methylation_level = max(0, self.methylation_level - 0.05)
        else:
            self.methylation_level = min(1, self.methylation_level + 0.05)

        # Probability of tumble based on nutrient and toxin gradients
        if toxin_gradient > 0:
            tumble_probability = 0.8  # High probability to tumble if moving towards toxins
        else:
            tumble_probability = 0.2 + (0.8 * self.methylation_level)  # Lower probability if moving towards nutrients

        if random.random() < tumble_probability or self.steps_since_tumble >= self.run_duration:
            self.direction_angle = random.uniform(0, 2 * math.pi)  # Random new direction
            self.run_duration = random.randint(5, 20)  # Randomize new run duration
            self.steps_since_tumble = 0
        else:
            self.steps_since_tumble += 1

        # Apply Brownian motion noise
        noise_angle = random.gauss(0, 0.1)  # Small random noise in direction
        self.direction_angle += noise_angle

        # Update position based on the current direction
        velocity = 1.0  # Speed of movement
        dx = velocity * math.cos(self.direction_angle)
        dy = velocity * math.sin(self.direction_angle)
        self.position += np.array([dx, dy])

        # Ensure position stays within boundaries
        self.position = np.clip(self.position, [0, 0], [self.medium_size[0] - 1, self.medium_size[1] - 1])

        # Update the previous concentrations for temporal sensing
        self.previous_nutrient = current_nutrient
        self.previous_toxin = current_toxin

def diffuse_and_decay(field, diffusion_rate, decay_rate, dt=1.0):
    # Applying diffusion equation with decay
    field = field + diffusion_rate * dt * (
        np.roll(field, 1, axis=0) +
        np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) +
        np.roll(field, -1, axis=1) -
        4 * field
    )
    field *= (1 - decay_rate * dt)  # Apply decay
    return field

# Simulation Parameters
medium_size = (200, 200)
num_steps = 1000

# Nutrient and Toxin Fields
nutrient_field = np.zeros(medium_size)
nutrient_field[150, 150] = 500  # Initial nutrient source
toxin_field = np.zeros(medium_size)
toxin_field[50, 50] = 500  # Initial toxin source

diffusion_rate_nutrient = 0.05
diffusion_rate_toxin = 0.03
decay_rate_nutrient = 0.005
decay_rate_toxin = 0.003

# # Initialize E. coli
# ecoli = EColi(position=[100, 100], sensing_range=5, medium_size=medium_size)

# # Run Simulation
# positions = []

# for step in range(num_steps):
#     # Diffuse and decay the nutrient and toxin fields
#     nutrient_field = diffuse_and_decay(nutrient_field, diffusion_rate_nutrient, decay_rate_nutrient)
#     toxin_field = diffuse_and_decay(toxin_field, diffusion_rate_toxin, decay_rate_toxin)
    
#     # Record the position of E. coli
#     positions.append(ecoli.position.copy())
    
#     # Sense nutrient and toxin gradients and move accordingly
#     ecoli.run_and_tumble(nutrient_field, toxin_field)

# # Plot E. coli movement path
# positions = np.array(positions)
# plt.imshow(nutrient_field, cmap='Greens', origin='lower', alpha=0.7, label='Nutrients')
# plt.imshow(toxin_field, cmap='Reds', origin='lower', alpha=0.3, label='Toxins')
# plt.plot(positions[:, 1], positions[:, 0], 'b', alpha=0.9, label='E. coli Path')
# plt.title("E. coli Movement Path in Medium with Nutrients and Toxins")
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.legend(["E. coli Path", "Nutrients", "Toxins"])
# plt.show()


