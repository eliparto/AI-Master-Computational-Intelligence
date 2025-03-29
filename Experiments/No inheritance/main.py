"""Main script for the FULL EXPERIMENTAL IMPLEMENTATION"""
"""
TO-DOs:
    - DE -> Combine T and C into one simulation
    - DE -> Place functions in DE class
    - Simulation starting seems to take up a lot of time
    - Only log necessary information
"""

import logging
import os
import sys
from time import localtime, perf_counter
from typing import Any
from tqdm import tqdm
import argparse
import plotille

import cma
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
from evaluator_body_script import Evaluator as Evaluator_body
from evaluator_brain_script import Evaluator as Evaluator_brain
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.evolution import ModularRobotEvolution
from revolve2.experimentation.evolution.abstract_elements import Reproducer, Selector, Learner
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.optimization.ea import population_management, selection
from revolve2.experimentation.rng import make_rng, seed_from_time


# Brain optimization variants
class BrainOptimizerCMA_ES(Learner):
    """Optimizer class (CMA-ES)"""
    
    def __init__(self) -> None:
        self
        
    def learn(
            self, population: Population, **kwargs: Any
            ) -> Population:
        """
        Generate individual robots from the population and optimize their weights
        
        :param population: Population of robots
        """
        # TODO: Find a way to extract/enter CPG weights
        logging.debug("\n\n### Starting learning loop ###\n\n")
        
        # Extract population and their fitnesses
        pop = population.individuals
        fit_old = [indiv.fitness for indiv in pop] # TODO: implement pop.individuals instead of using intermediate variable
        fit_new = []
        
        rng_seed = seed_from_time() % 2**32  # Cma seed must be smaller than 2**32.
        
        print("Controller learning:")
        for index in tqdm(range(len(pop)), leave = False):
            indiv = pop[index]
            # Generate robot (body and brain)
            body = indiv.genotype.develop().body
            active_hinges = body.find_modules_of_type(ActiveHinge)
    
            (
                cpg_network_structure,
                output_mapping,
            ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
            
            # Setup optimizer
            evaluator = Evaluator_brain(
            headless=True,
            num_simulators=config.NUM_SIMULATORS_BRAIN,
            cpg_network_structure=cpg_network_structure,
            body=body,
            output_mapping=output_mapping,
            )
            
            # Initial parameter values for the brain.
            num_joints = cpg_network_structure.num_connections
            if(num_joints > 1): # Robots with 0 or 1 joints cannot be optimized
                # TODO: Research different initialization values (He, Xavier etc.)
                initial_mean = num_joints * [0.5]
                logging.debug(f"Starting brain optimization on robot {index}")
                logging.debug(f"No. of joints: {num_joints}")
            
                # Initialize the cma optimizer.
                options = cma.CMAOptions()
                options.set("bounds", [-1.0, 1.0])
                options.set("seed", rng_seed)
                options.set("verb_disp", 0)
                # popsize calculation: 3*ln(4N) (N = no. of parameters/weights)
                #options.set("popsize", config.NUM_POPULATION_BRAIN_CMA) # TODO: ONLY FOR TESTING -> remove later!
                opt = cma.CMAEvolutionStrategy(initial_mean, config.INITIAL_STD, options)
                
                # Optimize brain -> export best weights and fitness
                for generation_index in tqdm(range(config.NUM_GENERATIONS_BRAIN_CMA), leave = False):
                    logging.debug(f"** Brain generation {generation_index + 1} / {config.NUM_GENERATIONS_BRAIN_CMA} **")
            
                    # Get the sampled solutions(parameters) from cma.
                    solutions = opt.ask()
                    # Evaluate them. Invert because fitness maximizes, but cma minimizes.
                    fitnesses = -evaluator.evaluate(solutions)
                    # Tell cma the fitnesses.
                    opt.tell(solutions, fitnesses)
                    logging.debug(f"{opt.result.xbest=} {opt.result.fbest=}")
            
                    # Increase the generation index counter.
                    generation_index += 1
                    
                logging.debug(f"\nFinished brain optimization on robot {index}\n")
                fit_new.append(-opt.result.fbest)
            
            else:
                logging.debug(f"Could not optimize robot {index}\n")
                fit_new.append(0.0) # Unoptimizable robot -> set fitness to zero

        #plot(fit_old, fit_new)
        logging.debug("\n\n### Learning done ###\n\n")
        
        # Output new population using Population() and Genotype()
        # TODO: DON'T update fitness to CMA solution if solution performs worse
        genotypes = [indiv.genotype for indiv in pop]
        fit = np.maximum(np.array(fit_old), np.array(fit_new)).tolist()
        consolePlot(fit)
        
        population = Population(
        individuals=[
            Individual(genotype=genotype, fitness=fitness)
            for genotype, fitness in zip(
                genotypes, fit, strict=True
                )
            ]
        )
        
        return population
 
class BrainOptimizerDE(Learner):
    """Optimizer class (CMA-ES)"""
    
    def __init__(self) -> None:
        self
        
    def learn(
            self, population: Population, **kwargs: Any
            ) -> Population:
        """
        Generate individual robots from the population and optimize their weights
        
        :param population: Population of robots
        """
        # TODO: Find a way to extract/enter CPG weights
        logging.debug("\n\n### Starting learning loop ###\n\n")
        
        # Extract population and their fitnesses
        pop = population.individuals
        fit_old = [indiv.fitness for indiv in pop] # TODO: implement pop.individuals instead of using intermediate variable
        fit_new = []
        
        for index, indiv in enumerate(pop):
            # Generate robot (body and brain)
            body = indiv.genotype.develop().body
            active_hinges = body.find_modules_of_type(ActiveHinge)
    
            (
                cpg_network_structure,
                output_mapping,
            ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
            
            # Setup optimizer
            evaluator = Evaluator_brain(
            headless=True,
            num_simulators=config.NUM_SIMULATORS_BRAIN,
            cpg_network_structure=cpg_network_structure,
            body=body,
            output_mapping=output_mapping,
            )
            
            # Initial parameter values for the brain.
            num_joints = cpg_network_structure.num_connections
            if(num_joints > 1): # Robots with 0 or 1 joints cannot be optimized
                # TODO: Research different initialization values (He, Xavier etc.)
                initial_mean = np.random.uniform(-1, 1, num_joints)
                logging.debug(f"Starting brain optimization on robot {index}")
                logging.debug(f"No. of joints: {num_joints}")
            
                # Sample target and candidate solutions
                sol_t, sol_c = DE(initial_mean)
                
                # Find best performers (next generation's targets)
                generation_index = 0
                while generation_index < config.NUM_GENERATIONS_BRAIN_DE:
                    logging.debug(f"** Brain generation {generation_index + 1} / {config.NUM_GENERATIONS_BRAIN_DE} **")
            
                    targets, max_fit = DE_optimize(sol_t, sol_c, evaluator)
                    
                    if(generation_index == config.NUM_POPULATION_BRAIN_DE - 1): break
                    sol_t, sol_c = DE(targets)
            
                    # Increase the generation index counter.
                    generation_index += 1
                
                fit_new.append(max_fit)
                logging.debug(f"\nFinished brain optimization on robot {index}\n")
            
            else:
                logging.debug(f"Could not optimize robot {index}\n")
                fit_new.append(0.0) # Unoptimizable robot -> set fitness to zero

        logging.debug("\n\n### Learning done ###\n\n")
        
        # Output new population using Population() and Genotype()
        # TODO: DON'T update fitness to CMA solution if solution performs worse
        genotypes = [indiv.genotype for indiv in pop]
        fit = np.maximum(fit_old, fit_new).tolist()
        
        population = Population(
        individuals=[
            Individual(genotype=genotype, fitness=fitness)
            for genotype, fitness in zip(
                genotypes, fit, strict=True
                )
            ]
        )
        
        return population   

def DE(vectors):
    """
    Performs DE (Differential Evolution) on an input vector of weights.
    
    :param vectors: Cadidate solution(s) to go through DE
    
    T ->    Target vectors:
            Add perturbation vectors P to copies of the input vector.
            T = T + P w/ P ~ N(o, sd)
    M ->    Mutation vectors:
            m_i = t_a + F(t_b - t_c) w/ a, b, and c some random indices.
    C ->    Crossover vectors:
            Every m_i gets a binary crossover mask with prob_cr to mix between m_i and t_i.
    C is outputted to be compared to T. The winning genes get passed on.
    """
    
    # Target vectors
    # Check if input is the initialized weights vector or a target matrix
    if len(vectors.shape) == 1:
        T = np.reshape(np.repeat(vectors, config.NUM_POPULATION_BRAIN_DE),
                       (vectors.shape[0], config.NUM_POPULATION_BRAIN_DE)).T
    else: T = vectors
    # Perturb the target vectors with perturbation vectors P
    P = np.random.normal(0, config.PERTURB_SD,
                         (config.NUM_POPULATION_BRAIN_DE, vectors.shape[0])) 
    T += P
    
    # Mutation
    # Mutant vectors
    M = np.zeros(vectors.shape[0])
    for i in range(config.NUM_POPULATION_BRAIN_DE):
        m_i = np.random.choice(np.arange(0, vectors.shape[0], 1), size = 3,
                                       replace = False) # Mutator indices
        m = T[m_i[0]] + config.F * (T[m_i[1]] - T[m_i[2]])
        M = np.vstack((M, m))
    M = M[1:]
    
    # Crossover
    cr_mask = np.random.choice([0,1], size = (config.NUM_POPULATION_BRAIN_DE, vectors.shape[0]),
                                              p = [1 - config.P_CR, config.P_CR])
    C = np.where(cr_mask == 1, M, T)
    
    return T, C
        
def DE_optimize(t, c, eval_class):
    """
    Selection mechanism for the next generation's target genes.

    :param t: Target vectors
    :param c: Candidate solutions
    """
    
    # Evaluate targets
    fit_t = eval_class.evaluate(t)
    fit_c = eval_class.evaluate(c)
    max_fitness = round(max(max(fit_t), max(fit_c)), 5)
    logging.debug(f"Best fitness: {max_fitness}")
    
    # Return the best performing weights
    targets = t[np.where(fit_t >= fit_c)[0]]
    targets = np.vstack((targets, c[np.where(fit_c > fit_t)[0]]))
    
    assert len(targets) == len(t), f"Length of target vectors is {len(targets)}. Should be {len(t)}"
    
    return targets, max_fitness

# Morphology optimization
class ParentSelector(Selector):
    """Selector class for parent selection."""

    rng: np.random.Generator
    offspring_size: int

    def __init__(self, offspring_size: int, rng: np.random.Generator) -> None:
        """
        Initialize the parent selector.

        :param offspring_size: The offspring size.
        :param rng: The rng generator.
        """
        self.offspring_size = offspring_size
        self.rng = rng

    def select(
        self, population: Population, **kwargs: Any
    ) -> tuple[npt.NDArray[np.int_], dict[str, Population]]:
        """
        Select the parents.

        :param population: The population of robots.
        :param kwargs: Other parameters.
        :return: The parent pairs.
        """
        return np.array(
            [
                selection.multiple_unique(
                    selection_size=2,
                    population=[
                        individual.genotype for individual in population.individuals
                    ],
                    fitnesses=[
                        individual.fitness for individual in population.individuals
                    ],
                    selection_function=lambda _, fitnesses: selection.tournament(
                        rng=self.rng, fitnesses=fitnesses, k=2
                    ),
                )
                for _ in range(self.offspring_size)
            ],
        ), {"parent_population": population}

class SurvivorSelector(Selector):
    """Selector class for survivor selection."""

    rng: np.random.Generator

    def __init__(self, rng: np.random.Generator) -> None:
        """
        Initialize the parent selector.

        :param rng: The rng generator.
        """
        self.rng = rng

    def select(
        self, population: Population, **kwargs: Any
    ) -> tuple[Population, dict[str, Any]]:
        """
        Select survivors using a tournament.

        :param population: The population the parents come from.
        :param kwargs: The offspring, with key 'offspring_population'.
        :returns: A newly created population.
        :raises ValueError: If the population is empty.
        """
        offspring = kwargs.get("children")
        offspring_fitness = kwargs.get("child_task_performance")
        if offspring is None or offspring_fitness is None:
            raise ValueError(
                "No offspring was passed with positional argument 'children' and / or 'child_task_performance'."
            )

        original_survivors, offspring_survivors = population_management.steady_state(
            old_genotypes=[i.genotype for i in population.individuals],
            old_fitnesses=[i.fitness for i in population.individuals],
            new_genotypes=offspring,
            new_fitnesses=offspring_fitness,
            selection_function=lambda n, genotypes, fitnesses: selection.multiple_unique(
                selection_size=n,
                population=genotypes,
                fitnesses=fitnesses,
                selection_function=lambda _, fitnesses: selection.tournament(
                    rng=self.rng, fitnesses=fitnesses, k=2
                ),
            ),
        )

        return (
            Population(
                individuals=[
                    Individual(
                        genotype=population.individuals[i].genotype,
                        fitness=population.individuals[i].fitness,
                    )
                    for i in original_survivors
                ]
                + [
                    Individual(
                        genotype=offspring[i],
                        fitness=offspring_fitness[i],
                    )
                    for i in offspring_survivors
                ]
            ),
            {},
        )

class CrossoverReproducer(Reproducer):
    """A simple crossover reproducer using multineat."""

    rng: np.random.Generator
    innov_db_body: multineat.InnovationDatabase
    innov_db_brain: multineat.InnovationDatabase

    def __init__(
        self,
        rng: np.random.Generator,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
    ):
        """
        Initialize the reproducer.

        :param rng: The ranfom generator.
        :param innov_db_body: The innovation database for the body.
        :param innov_db_brain: The innovation database for the brain.
        """
        self.rng = rng
        self.innov_db_body = innov_db_body
        self.innov_db_brain = innov_db_brain

    def reproduce(
        self, population: npt.NDArray[np.int_], **kwargs: Any
    ) -> list[Genotype]:
        """
        Reproduce the population by crossover.

        :param population: The parent pairs.
        :param kwargs: Additional keyword arguments.
        :return: The genotypes of the children.
        :raises ValueError: If the parent population is not passed as a kwarg `parent_population`.
        """
        parent_population: Population | None = kwargs.get("parent_population")
        if parent_population is None:
            raise ValueError("No parent population given.")

        offspring_genotypes = [
            Genotype.crossover(
                parent_population.individuals[parent1_i].genotype,
                parent_population.individuals[parent2_i].genotype,
                self.rng,
            ).mutate(self.innov_db_body, self.innov_db_brain, self.rng)
            for parent1_i, parent2_i in population
        ]
        return offspring_genotypes

            
# Experiment
def run_experiment(dbengine: Engine) -> None:
    """
    Run an experiment.

    :param dbengine: An openened database with matching initialize database structure.
    """
    logging.debug("----------------")
    logging.debug("Start experiment")

    # Set up the random number generator.
    rng_seed = seed_from_time()
    rng = make_rng(rng_seed)

    # Create and save the experiment instance.
    experiment = Experiment(rng_seed=rng_seed)
    logging.debug("Saving experiment configuration.")
    with Session(dbengine) as session:
        session.add(experiment)
        session.commit()

    # CPPN innovation databases.
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    """
    Here we initialize the components used for the evolutionary process:
    - evaluator_body: Allows us to evaluate a population of modular robots.
    - parent_selector: Allows us to select parents from a population of modular robots.
    - survivor_selector: Allows us to select survivors from a population.
    - crossover_reproducer: Allows us to generate offspring from parents.
    - modular_robot_evolution: The evolutionary process as a object that can be iterated.
    - learner: Allows for the individual robots in the population to learn.
    """
    evaluator_body = Evaluator_body(headless=True, num_simulators=config.NUM_SIMULATORS_BODY)
    parent_selector = ParentSelector(offspring_size=config.OFFSPRING_SIZE, rng=rng)
    survivor_selector = SurvivorSelector(rng=rng)
    crossover_reproducer = CrossoverReproducer(
        rng=rng, innov_db_body=innov_db_body, innov_db_brain=innov_db_brain
    )
    learner = BrainOptimizerCMA_ES()

    modular_robot_evolution = ModularRobotEvolution(
        parent_selection=parent_selector,
        survivor_selection=survivor_selector,
        evaluator=evaluator_body,
        reproducer=crossover_reproducer,
        learner=learner
    )
    
    # TODO: Brain optimizer has to be started in a loop for every robot
    # TODO: See if brain optimizer can be implemented into the modular_robot class

    # Create an initial population, as we cant start from nothing.
    logging.debug("Generating initial population.")
    initial_genotypes = [
        Genotype.random(
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            rng=rng,
        )
        for _ in range(config.POPULATION_SIZE_BODY)
    ]

    # Evaluate the initial population.
    logging.debug("Evaluating initial population.")
    initial_fitnesses = evaluator_body.evaluate(initial_genotypes)

    # Create a population of individuals, combining genotype with fitness.
    population = Population(
        individuals=[
            Individual(genotype=genotype, fitness=fitness)
            for genotype, fitness in zip(
                initial_genotypes, initial_fitnesses, strict=True
            )
        ]
    )

    # Finish the zeroth generation and save it to the database.
    generation = Generation(
        experiment=experiment, generation_index=0, population=population
    )
    save_to_db(dbengine, generation)

    # Start the actual optimization process/evolutionary loop
    # Optimize brain -> Optimize body -> LOOP
    logging.debug("Start optimization process.")
    print("Body generations:")
    for it in tqdm(range(config.NUM_GENERATIONS_BODY), leave = False):
        generation.generation_index = it
        logging.debug(
            f"\n\n### Morphology generation {generation.generation_index + 1} / {config.NUM_GENERATIONS_BODY} ###\n\n"
        )

        # Here we iterate the evolutionary process using the step.
        population = modular_robot_evolution.step(population)

        # Make it all into a generation and save it to the database.
        generation = Generation(
            experiment=experiment,
            generation_index=generation.generation_index + 1,
            population=population,
        )
        save_to_db(dbengine, generation)


def main() -> None:
    # Check for passed arguments
    parser = argparse.ArgumentParser(description="Simulate evolution.")
    parser.add_argument("-r", action="store_true", help="add to remove the prior log.")
    parser.add_argument("-name", type=str, help="Specify the database filename.")
    args = parser.parse_args()
    
    if args.name:   
        dbName = args.name + ".sqlite"
        logName = args.name + "_log.txt"
    else: 
        dbName = config.DATABASE_FILE 
        logName = "log.txt"
        
    if args.r: os.remove(dbName)
    
    """Run the program."""
    # Set up logging.
    setup_logging(file_name=logName, level = 30)
    logConfig()

    # Open the database, only if it does not already exists.
    dbengine = open_database_sqlite(
        dbName, open_method=OpenMethod.NOT_EXISTS_AND_CREATE
    )
    # Create the structure of the database.
    Base.metadata.create_all(dbengine)

    # Run the experiment several times.
    for _ in range(config.NUM_REPETITIONS_BODY):
        timeStart = int(perf_counter())
        run_experiment(dbengine)
        timeDelta = (int(perf_counter()) - timeStart) // 60
        logging.info(f"\n*** Run took {timeDelta} minutes ***\n")

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

def consolePlot(fit):
    """
    Plot all fitnesses after learning
    """
    x = np.arange(0, len(fit), 1)
    fig = plotille.Figure()
    fig.set_x_limits(0, len(fit))
    fig.width = 60
    fig.height = 30
    fig.plot(x, fit, interp = "linear", lc = "bright_yellow_old")
    fig.plot(x, sorted(fit), interp = "linear", lc = "magenta")
    
    print(fig.show())


def logConfig():
    """
    Print the 'config.py' parameters to the log.
    """
    logging.info("## BODY PARAMETERS")
    logging.info(f"Experiment repetitions:\t\t{config.NUM_REPETITIONS_BODY}")
    logging.info(f"No. of simulators:\t\t{config.NUM_SIMULATORS_BODY}")
    logging.info(f"No. of generations:\t\t{config.NUM_GENERATIONS_BODY}")
    logging.info(f"Population size:\t\t{config.POPULATION_SIZE_BODY}")
    logging.info(f"Offspring size:\t\t\t{config.OFFSPRING_SIZE}")

    logging.info("\n\n## CMA PARAMETERS")
    logging.info(f"Experiment repetitions:\t\t{config.NUM_REPETITIONS_BRAIN}")
    logging.info(f"No. of simulators:\t\t{config.NUM_SIMULATORS_BRAIN}")
    logging.info(f"No. of generations:\t\t{config.NUM_GENERATIONS_BRAIN_CMA}")
    logging.info(f"Population size:\t\t{config.NUM_POPULATION_BRAIN_CMA}")
    logging.info(f"INITIAL SD:\t\t\t{config.INITIAL_STD}")
    
    logging.info("\n\n## DE PARAMETERS")
    logging.info(f"Experiment repetitions:\t\t{config.NUM_REPETITIONS_BRAIN}")
    logging.info(f"No. of simulators:\t\t{config.NUM_SIMULATORS_BRAIN}")
    logging.info(f"No. of generations:\t\t{config.NUM_GENERATIONS_BRAIN_DE}")
    logging.info(f"Population size:\t\t{config.NUM_POPULATION_BRAIN_DE}")
    logging.info(f"PERTURB SD:\t\t\t{config.PERTURB_SD}")
    logging.info(f"F-value:\t\t\t{config.F}")  

if __name__ == "__main__":
    main()
