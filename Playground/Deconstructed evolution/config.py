"""
Configuration parameters
"""

# Body
DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS_BODY = 1 # No. of repetitions of the morphology optimization process
POPULATION_SIZE_BODY = 20
OFFSPRING_SIZE = 3
NUM_GENERATIONS_BODY = 1 # No. of body optimization generations

# Differential Evolution parameters
NUM_SIMULATORS_BRAIN = 6
NUM_GENERATIONS_BRAIN = 3
NUM_POPULATION_BRAIN = 5
PERTURB_SD_MOD = 2
P_CR = 0.7 # [0,1]
P_MU = 0.5 # [0,1]
F = 1.5 # [0,2]
