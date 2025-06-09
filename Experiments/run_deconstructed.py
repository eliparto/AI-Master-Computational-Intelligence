""" Deconstructed generational step for variable exploration with object importing """

import logging
import os

import config
import multineat
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

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.evolution import ModularRobotEvolution
from revolve2.experimentation.rng import make_rng, seed_from_time

from BodyCheck import BodyCheck
from BrainOptimizer import BrainOptimizerDE
from ModularEvolution import ParentSelector, CrossoverReproducer, SurvivorSelector

# Database
def save_to_db(dbengine: Engine, generation: Generation) -> None:
    """
    Save the current generation to the database.

    :param dbengine: The database engine.
    :param generation: The current generation.
    """
    logging.debug("Saving generation.")
    with Session(dbengine, expire_on_commit=False) as session:
        session.add(generation)
        session.commit()
        
# Setup
# Database setup
# Try to remove old database
try:
    os.remove("test_db.sqlite")
except:
    pass

# Open the database, only if it does not already exists.
dbengine = open_database_sqlite(
    "test_db.sqlite", open_method=OpenMethod.NOT_EXISTS_AND_CREATE
)

# Create the structure of the database.
Base.metadata.create_all(dbengine)

# Experiment setup:
rng_seed = seed_from_time()
rng = make_rng(rng_seed)
bounds = (-5,5)

# Create and save the experiment instance.
experiment = Experiment(rng_seed=rng_seed)
logging.debug("Saving experiment configuration.")
with Session(dbengine) as session:
    session.add(experiment)
    session.commit()
    
innov_db_body = multineat.InnovationDatabase()
innov_db_brain = multineat.InnovationDatabase()

learner = BrainOptimizerDE(bounds, use_state_reset=True, inherit=False)    
parent_selector = ParentSelector(offspring_size=config.OFFSPRING_SIZE, rng=rng)
survivor_selector = SurvivorSelector(rng=rng)
crossover_reproducer = CrossoverReproducer(
    rng=rng, innov_db_body=innov_db_body, innov_db_brain=innov_db_brain
)
morpho = BodyCheck()

modular_robot_evolution = ModularRobotEvolution(
    parent_selection=parent_selector,
    survivor_selection=survivor_selector,
    reproducer=crossover_reproducer,
    learner=learner,
    morpho=morpho
)

# Generate the initial population's genotypes
initial_genotypes = [
    Genotype.random(
        innov_db_body=innov_db_body,
        innov_db_brain=innov_db_brain,
        rng=rng,
    )
    for _ in range(config.POPULATION_SIZE_BODY)
]

# Create the initial population (0 fitness and no solution)
population = Population(
    individuals=[
        Individual(genotype=genotype, fitness=0.0, nose = -1, solutions=[]
                   )
        for genotype in initial_genotypes
        ]
    )

# Finish the zeroth generation and save it to the database.
generation = Generation(
    experiment=experiment, generation_index=0, population=population,
)
save_to_db(dbengine, generation)
