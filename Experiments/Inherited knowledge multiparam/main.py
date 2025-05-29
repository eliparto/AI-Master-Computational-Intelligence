""" Main script for the FULL EXPERIMENTAL IMPLEMENTATION (MULTIPARAM OPTIMIZATION) """

import os
from tqdm import tqdm
import argparse
from consolePlot import consolePlot

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

from ModularEvolution import ParentSelector, SurvivorSelector, CrossoverReproducer
from BrainOptimizer import BrainOptimizerDE
from BodyCheck import BodyCheck
from writeOut import writeSetup
       
# Experiment
def run_experiment(
        dbengine: Engine, plot: bool, use_state_reset: bool
        ) -> None:
    """
    Run an experiment.

    :param dbengine: An opened database with matching initialize database structure.
    :param plot: Bool to toggle per-generaration in-console fitness plotting.
    :param newState: Bool to toggle CPGs with state-array resetting when changing actions.
    """
    # Set up and create experiment instance
    rng_seed = seed_from_time()
    rng = make_rng(rng_seed)
    experiment = Experiment(rng_seed=rng_seed)
    with Session(dbengine) as session:
        session.add(experiment)
        session.commit()

    # CPPN innovation databases.
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    """
    Here we initialize the components used for the evolutionary process:
    - learner: Allows for the individual robots in the population to learn.
    - parent_selector: Allows us to select parents from a population of modular robots.
    - survivor_selector: Allows us to select survivors from a population.
    - crossover_reproducer: Allows us to generate offspring from parents.
    - modular_robot_evolution: The evolutionary process as a object that can be iterated.
    - morpho: Morphology analyzer used for e.g. finding frontal orientation and visualization.
    """
    morpho = BodyCheck()
    learner = BrainOptimizerDE(
        bounds=config.BOUNDS, use_state_reset=use_state_reset
    )    
    parent_selector = ParentSelector(offspring_size=config.OFFSPRING_SIZE, rng=rng)
    survivor_selector = SurvivorSelector(rng=rng)
    crossover_reproducer = CrossoverReproducer(
        rng=rng, innov_db_body=innov_db_body, innov_db_brain=innov_db_brain
    )
    
    modular_robot_evolution = ModularRobotEvolution(
        parent_selection=parent_selector,
        survivor_selection=survivor_selector,
        reproducer=crossover_reproducer,
        learner=learner,
        morpho=morpho,
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
    # Train the initial population -> Start by generating solutions and finding nose orientations
    print("\nPopulation initialized.\nTraining initial population...\n")
    population = morpho.findNose(population)
    population = learner.initialSolutions(population)
    population = learner.learn(population)
    print("Initial population ready.")

    # Finish the zeroth generation and save it to the database.
    generation = Generation(
        experiment=experiment, generation_index=0, population=population,
    )
    save_to_db(dbengine, generation)

    # Start the actual optimization process/evolutionary loop
    print("\nOptimizing...\n")
    for it in tqdm(range(config.NUM_GENERATIONS_BODY), leave = True,
                   position = 0):
        generation.generation_index = it

        if plot: # Plot fitnesses in console per generation
            prev_fit = [p.fitness for p in population.individuals]
            population = modular_robot_evolution.step(population)
            curr_fit = [p.fitness for p in population.individuals]
            consolePlot(prev_fit, curr_fit)

        # Make it all into a generation and save it to the database.
        generation = Generation(
            experiment=experiment,
            generation_index=generation.generation_index + 1,
            population=population,
        )
        save_to_db(dbengine, generation)

def main() -> None:
    # Check for passed arguments
    parser = argparse.ArgumentParser(
        description=config.DESC,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-name", type=str, help="Specify the database filename.")
    parser.add_argument("-r", action="store_true", help="Remove the prior log.")
    parser.add_argument("-p", action="store_true", help="Plot fitnesses per generation")
    parser.add_argument("-e", action="store_true", help="Add to simulate using state-resetting.")

    args = parser.parse_args()
    
    # Check if the databases folder is present
    if os.path.exists("Databases") == False: os.mkdir("Databases")
    
    if args.name:   
        dbName = "Databases/" + args.name + ".sqlite"
        writeSetup(args.name, args.e)
    else: 
        dbName = "Databases/" + config.DATABASE_FILE 
        
    if args.r: 
        try:
            os.remove(dbName)
        except:
            pass
    
    """Run the program."""
    # Open the database, only if it does not already exists.
    dbengine = open_database_sqlite(
        dbName, open_method=OpenMethod.NOT_EXISTS_AND_CREATE
    )
    # Create the structure of the database.
    Base.metadata.create_all(dbengine)

    # Run the experiment several times.
    for i in range(config.NUM_REPETITIONS_BODY):
        print(f"Experiment no. {i+1}/{config.NUM_REPETITIONS_BODY}:")
        run_experiment(dbengine, args.p, args.e)

def save_to_db(dbengine: Engine, generation: Generation) -> None:
    """
    Save the current generation to the database.

    :param dbengine: The database engine.
    :param generation: The current generation.
    """
    with Session(dbengine, expire_on_commit=False) as session:
        session.add(generation)
        session.commit()

if __name__ == "__main__":
    main()
