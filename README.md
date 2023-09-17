# PyColi
Simple model of E. coli in a nutrient soup.

I cover the results of simulations in my substack article [Life Without a Brain](https://open.substack.com/pub/rememberization/p/life-without-a-brain?r=1pbu6n&utm_medium=ios&utm_campaign=post).  

# Usage
* Clone the repository
* Install the requirements with `pip3 install -r requirements.txt`
* Run `python3 main.py` to run the simulation

To change the parameters of the simulation, update via the command line arguments:

* `--time_steps` to change the number of simulation steps
* `--grid_size_x` and `--grid_size_y` to change the size of the grid
* `--move_function` to change the movement function of the bacteria. See Move section below
* `--placement` to change the initial placement of the bacteria (random or center)

Example: `python3 main.py --grid_size_x 100 --grid_size_y 100 --consumption_rate 0.1 --placement center`

The script will run the simulation and save the movement in a gif

At some point I will try and see if I can visually replicate this video of E. coli in a nutrient soup: https://youtu.be/F6QMU3KD7zw?si=3CpUbphJo2pg7ATN

## Move Functions

* `random_walk` - Randomly move in any direction
* `biased_random_walk` - Randomly move in any direction, but with a bias towards higher nutrient concentrations
* `biased_random_walk_with_memory` - biased random walk with memory of the previous nutrient concentration and additional bias to higher concentrations

# Results

## Random Walk
![](docs/random_walk.gif]
