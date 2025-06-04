""" Print out parameters """

import config
import logging

def logConfig():
    logging.info("## BODY PARAMETERS")
    logging.info(f"Experiment repetitions:\t\t{config.NUM_REPETITIONS_BODY}")
    logging.info(f"No. of generations:\t\t\t{config.NUM_GENERATIONS_BODY}")
    logging.info(f"Population size:\t\t\t\t{config.POPULATION_SIZE_BODY}")
    logging.info(f"Offspring size:\t\t\t\t{config.OFFSPRING_SIZE}")
    
    logging.info("\n\n## DE PARAMETERS")
    logging.info(f"No. of simulators:\t\t\t{config.NUM_SIMULATORS_BRAIN}")
    logging.info(f"No. of generations:\t\t\t{config.NUM_GENERATIONS_BRAIN}")
    logging.info(f"Population size:\t\t\t\t{config.NUM_POPULATION_BRAIN}")
    logging.info(f"Perutrbation SD:\t\t\t\t{config.PERTURB_SD_MOD}")
    logging.info(f"Crossover rate:\t\t\t\t{config.P_CR}")
    logging.info(f"Mutation rate:\t\t\t\t{config.P_MU}")    
    logging.info(f"F-value:\t\t\t\t\t\t{config.F}")
    logging.info("\n########################################\n")