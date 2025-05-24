""" Write experiment setup/results to file. """

import os
import config

txt = (
       "EXPERIMENTAL SETUP:\n"
       f"TARGET XY BOUNDS:\t\t{config.BOUNDS}\n"
       f"NO. OF EXPERIMENT RUNS:\t{config.NUM_REPETITIONS_BODY}\n"
       f"NO. OF SIMULATORS:\t\t{config.NUM_SIMULATORS_BRAIN}\n\n"
       "BODIES:\n"
       f"POPULATION SIZE:\t\t{config.POPULATION_SIZE_BODY}\n"
       f"OFFSPRING SIZE\t\t\t{config.OFFSPRING_SIZE}\n"
       f"NO. OF GENERATIONS:\t\t{config.NUM_GENERATIONS_BODY}\n\n"
       "BRAINS:\n"
       f"SOLS POPULATION SIZE:\t{config.NUM_GENERATIONS_BRAIN}\n"
       f"NO. OF GENERATIONS:\t\t{config.NUM_POPULATION_BRAIN}\n"
       f"CROSSOVER PROB:\t\t\t{config.P_CR}\n"
       f"MUTATION PROB:\t\t\t{config.P_MU}\n"
       f"F-FACTOR:\t\t\t\t{config.F}\n"
       )

def writeSetup(fileName):
    fileName = "Databases/" + fileName + ".txt"
    with open(fileName, "w") as f:
        f.write(txt)
        
def writeResults(text, fileName):
    pass # Not implemented