"""Configuration parameters"""

# Body
DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS_BODY = 1 # No. of repetitions of the morphology optimization process
NUM_SIMULATORS_BODY = 2
POPULATION_SIZE_BODY = 20
OFFSPRING_SIZE = 10
NUM_GENERATIONS_BODY = 15 # No. of body optimization generations

# Brain
DATABASE_FILE = "database_brain.sqlite"
NUM_REPETITIONS_BRAIN = 1 # Repetitions of the optimizer
NUM_SIMULATORS_BRAIN = 1
INITIAL_STD = 0.5
NUM_GENERATIONS_BRAIN_CMA = 10 #10 # No. of generations that the optimizer runs
NUM_POPULATION_BRAIN_CMA = 5 #5 # No. of genome candidates going through optimization. Setup in options.set("popsize", VALUE) TODO: remove

# Differential Evolution parameters
NUM_GENERATIONS_BRAIN_DE = 10
NUM_POPULATION_BRAIN_DE = 10
PERTURB_SD = 0.25
P_CR = 0.9 # [0,1]
P_MU = 0.5 # [0,1]
F = 1.0 # [0,2]

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