""" Write experiment setup/results to file. """

import os
import config

txt = (
       "EXPERIMENTAL SETUP:\n"
       f"TARGET COORDS:\t\t\t{config.TARGETS[0].tolist(),config.TARGETS[1].tolist(),config.TARGETS[2].tolist()}\n"
       f"SIMULATION TIME:\t\t{config.SIM_TIME}\n"
       f"NO. OF EXPERIMENT RUNS:\t\t{config.NUM_REPETITIONS_BODY}\n"
       f"NO. OF SIMULATORS:\t\t{config.NUM_SIMULATORS_BRAIN}\n\n"
       "BODIES:\n"
       f"POPULATION SIZE:\t\t{config.POPULATION_SIZE_BODY}\n"
       f"OFFSPRING SIZE\t\t\t{config.OFFSPRING_SIZE}\n"
       f"NO. OF GENERATIONS:\t\t{config.NUM_GENERATIONS_BODY}\n\n"
       "BRAINS:\n"
       f"SOLS POPULATION SIZE:\t\t{config.NUM_GENERATIONS_BRAIN}\n"
       f"NO. OF GENERATIONS:\t\t{config.NUM_POPULATION_BRAIN}\n"
       f"CROSSOVER PROB:\t\t\t{config.P_CR}\n"
       f"MUTATION PROB:\t\t\t{config.P_MU}\n"
       f"F-FACTOR:\t\t\t{config.F}\n"
       )

def writeSetup(fileName):
    fileName = "Databases/" + fileName + ".txt"
    with open(fileName, "w") as f:
        f.write(txt)
        
def writeResults(text, fileName):
    pass # Not implemented