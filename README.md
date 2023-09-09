# PyColi
Simple model of E. coli in a nutrient soup

# Usage
* Clone the repository
* Install the requirements with `pip3 install -r requirements.txt`
* Run `python3 main.py` to run the simulation

To change the parameters of the simulation, update via the command line arguments:

* `--grid_size_x` and `--grid_size_y` to change the size of the grid
* `--consumption_rate` to change the rate at which the bacteria consume the nutrient
* `--placement` to change the initial placement of the bacteria (random or center)

Example: `python3 main.py --grid_size_x 100 --grid_size_y 100 --consumption_rate 0.1 --placement center`

The script will run the simulation and save the movement in a gif

At some point I will try and see if I can visually replicate this video of E. coli in a nutrient soup: https://youtu.be/F6QMU3KD7zw?si=3CpUbphJo2pg7ATN