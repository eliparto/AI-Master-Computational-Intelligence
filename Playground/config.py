"""Configuration parameters"""

# Body
DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS_BODY = 1 # No. of repetitions of the morphology optimization process
NUM_SIMULATORS_BODY = 1
POPULATION_SIZE_BODY = 8
OFFSPRING_SIZE = 4
NUM_GENERATIONS_BODY = 5 # No. of body optimization generations

# CMA-ES parameters
DATABASE_FILE = "database_brain.sqlite"
NUM_REPETITIONS_BRAIN = 1 # Repetitions of the optimizer
NUM_SIMULATORS_BRAIN = 1
INITIAL_STD = 0.5
NUM_GENERATIONS_BRAIN_CMA = 10 #10 # No. of generations that the optimizer runs
NUM_POPULATION_BRAIN_CMA = 5 #5 # No. of genome candidates going through optimization. Setup in options.set("popsize", VALUE) TODO: remove

# Differential Evolution parameters
NUM_GENERATIONS_BRAIN_DE = 5
NUM_POPULATION_BRAIN_DE = 8
PERTURB_SD_MOD = 2
P_CR = 0.7 # [0,1]
P_MU = 0.5 # [0,1]
F = 1.25 # [0,2]

# ORIGINAL PARAMETERS:
# # Body
# DATABASE_FILE = "database.sqlite"
# NUM_REPETITIONS = 5
# NUM_SIMULATORS = 8
# POPULATION_SIZE = 100
# OFFSPRING_SIZE = 50
# NUM_GENERATIONS = 10

# # Brain
# DATABASE_FILE = "database.sqlite"
# NUM_REPETITIONS = 4
# NUM_SIMULATORS = 4
# INITIAL_STD = 0.5
# NUM_GENERATIONS = 5