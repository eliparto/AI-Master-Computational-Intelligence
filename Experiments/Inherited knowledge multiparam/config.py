"""
Configuration parameters
"""
import numpy as np

# Experiment
BOUNDS = (-5,5) # Target xy-coordinate bounds
SIM_TIME = 80 # Simulation time
TARGETS = np.array(
    [[1,1],
     [2,-2],
     [-2,-1]]
    )

DESC = (
        "Simulate evolution with inherited knowledge.\n"
        "Don't forget to:\n"
        " - Check parameters in config.py.\n"
        " - Set the number of simulators to maximally utilize CPU for multithreaded simulation.\n(Recommend no. of cores minus 2)s\n"
        )

# Body
DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS_BODY = 3 # No. of repetitions of the morphology optimization process
POPULATION_SIZE_BODY = 40
OFFSPRING_SIZE = 20
NUM_GENERATIONS_BODY = 10 # No. of body optimization generations

# Differential Evolution parameters
NUM_SIMULATORS_BRAIN = 6
NUM_GENERATIONS_BRAIN = 8
NUM_POPULATION_BRAIN = 10
PERTURB_SD_MOD = 2
P_CR = 0.7 # [0,1]
P_MU = 0.5 # [0,1]
F = 1.25 # [0,2]
UNIF_SAMPLE = 0.7

