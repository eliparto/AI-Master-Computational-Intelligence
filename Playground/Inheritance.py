""" CPG Network and knowledge inheritance experimentation """

import logging
from typing import Any
from tqdm import tqdm

import config
import multineat
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from database_components import (
    Base,
    Experiment,
    Generation,
    Genotype,
    Individual,
    Population,
)

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)

from evaluator_body_script import Evaluator as Evaluator_body

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.evolution import ModularRobotEvolution
from revolve2.experimentation.evolution.abstract_elements import Reproducer, Selector, Learner
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.optimization.ea import population_management, selection
from revolve2.experimentation.rng import make_rng, seed_from_time

# Setup
rng_seed = seed_from_time()
rng = make_rng(rng_seed)
experiment = Experiment(rng_seed=rng_seed)
innov_db_body = multineat.InnovationDatabase()
innov_db_brain = multineat.InnovationDatabase()
evaluator_body = Evaluator_body(headless=True, num_simulators=config.NUM_SIMULATORS_BODY)

# Initial population
initial_genotypes = [
    Genotype.random(
        innov_db_body=innov_db_body,
        innov_db_brain=innov_db_brain,
        rng=rng,
    )
    for _ in range(config.POPULATION_SIZE_BODY)
]

initial_fitnesses = evaluator_body.evaluate(initial_genotypes)

# Create a population of individuals, combining genotype with fitness.
#p_sol = np.random.normal(size = 4).tolist()
population = Population(
    individuals=[ #TODO: p_sol added
        Individual(genotype=genotype, fitness=fitness, p_sol=0)
        for genotype, fitness in zip(
            initial_genotypes, initial_fitnesses, strict=True
        )
    ]
)

# Finish the zeroth generation and save it to the database.
generation = Generation(
    experiment=experiment, generation_index=0, population=population
)

p_sol = np.random.normal(size = 4).tolist()
print("Test solution:", p_sol)

for pop in population.individuals:
    pop.p_sol = p_sol

print("Reading p_sol values:")
for pop in population.individuals:
    print(pop.p_sol)



