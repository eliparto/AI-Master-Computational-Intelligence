""" Deconsturcted generational step for variable exploration """

import logging
import os
import json
from typing import Any
from tqdm import tqdm
import argparse

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

# Brain optimizer
class BrainOptimizerDE(Learner):
    """Optimizer class (DE)"""
    
    def __init__(self) -> None:
        self
        
    def learn(
            self, population: Population, **kwargs: Any) -> Population:
        """
        Generate individual robots from the population and optimize their weights
        
        :param population: Population to go through DE.
        """
        
        # Generate children bodies and brains
        bodies, brains, solution_sizes = self.setupLearner(population)
        
        # Reformat solution vectors to the correct sizes
        population = self.setSolutionSizes(population, solution_sizes)
        
        print("Optimizing brains")
        for idx, body in enumerate(tqdm(bodies, leave = False)):
            # Setup optimizer
            cpg_network_structure, output_mapping = brains[idx]
            
            # Only optimize robots with at least 2 joints
            if cpg_network_structure.num_connections > 1:
                evaluator = Evaluator_brain(
                headless=True,
                num_simulators=config.NUM_SIMULATORS_BRAIN,
                cpg_network_structure=cpg_network_structure,
                body=body,
                output_mapping=output_mapping,
                )
                
                # # Sample target and candidate solutions from stored parent solution
                # sol_t, sol_c = self.DE(population.individuals[idx].solution)

                # for gen in tqdm(range(config.NUM_GENERATIONS_BRAIN - 1),
                #                 leave = False):
                #     targets, _ = self.DE_optimize(sol_t, sol_c, evaluator)
                #     sol_t, sol_c = self.DE(targets)
                    
                # # Update fitness and solution
                # targets, max_fit = self.DE_optimize(sol_t, sol_c, evaluator)
                # population.individuals[idx].solution = targets[0].tolist()
                # population.individuals[idx].fitness = max_fit
                
            else:
                population.individuals[idx].fitness_forward = 0.0
                population.individuals[idx].fitness_rot_l = 0.0
                population.individuals[idx].fitness_rot_r = 0.0
                
            # Calculate individual's total fitness
            population.individuals[idx].fitness_total = (
                population.individuals[idx].fitness_forward + \
                population.individuals[idx].fitness_forward + \
                population.individuals[idx].fitness_forward
                )
                
        return population
    
    def getFitness(
            self, fit_type: int, evaluator) -> None: # TODO: fix type casting
        """
        Train and learn fitness for specific action (moving forward or rotating).
        :fit_type:  Integer stating type of training:
                    1 -> Moving forward
                    2 -> Rotating left
                    3 -> Rotating right
        """
        if fit_type == 1: sol_t, sol_c = self.DE(
                population.individuals[idx].solution_forward)
        elif fit_type == 2: sol_t, sol_c = self.DE(
                population.individuals[idx].solution_rot_l)
        else: sol_t, sol_c = self.DE(
            population.individuals[idx].solution_rot_r)
        
        for gen in tqdm(range(config.NUM_GENERATIONS_BRAIN - 1),
                        leave = False):
            targets, _ = self.DE_optimize(sol_t, sol_c, evaluator)
            sol_t, sol_c = self.DE(targets)
            
    
    def DE(
            self, vectors):
        """
        Generates target and candidate vectors for Differential Evolution).
        
        :param vectors: Cadidate solution(s) to go through DE. Can be 1D list or 2D array.
        
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
        vectors = np.array(vectors) # Reformat
        if vectors.ndim == 1:
            T = np.reshape(np.repeat(
                vectors, config.NUM_POPULATION_BRAIN),
                (len(vectors), config.NUM_POPULATION_BRAIN)
                ).T
        else: T = vectors

        # Perturb the target vectors with perturbation vectors P
        P = np.random.normal(0, np.std(T) / config.PERTURB_SD_MOD, size = T.shape)
        T += P
        
        # Mutation
        m_1, m_2, m_3 = self.mutationIndices(len(T))
        M = T[m_1] + config.F * (T[m_2] - T[m_3])
           
        # Crossover (use binary mask to decide if T or C is used)
        cr_mask = np.random.choice(
            [0,1], size = T.shape, p = [1 - config.P_CR, config.P_CR]
            )
        C = np.where(cr_mask == 1, M, T)
        
        return T, C
    
    def DE_optimize(
            self, T: npt.NDArray[np.float_], C: npt.NDArray[np.float_],
            fit_type: int, evaluator):
        """
        Compare target vectors with candidate vectors for the next generation.
    
        :param T: Target vectors.
        :param C: Candidate solutions.
        :fit_type: Integer stating movement type to optimize.
        :evaluator: Fitness evaluator class.
        """
        
        logging.debug("DE: Comparing targets with candidates")
        # Evaluate targets
        solutions = np.vstack((T, C))
        if fit_type == 1: fitnesses, _, _, betas = evaluator.evaluate(solutions)
        elif fit_type == 2: _, fitnesses, _, betas = evaluator.evaluate(solutions)
        else: _, _, fitnesses, betas = evaluator.evaluate(solutions)
        
        # Sort targets by fitness (high to low)
        sort_idx = np.flip(np.argsort(fitnesses))
        solutions = solutions[sort_idx]
        
        return solutions[:config.NUM_POPULATION_BRAIN], max(fitnesses)
    
    def mutationIndices(
            self, t_pop) -> npt.NDArray[np.int_]:
        """
        Generate the indices for the mutation arrays.

        :param t_pop: No. of target vectors to choose from.
        """
        assert t_pop > 3, f"Need at least 4 vectors to choose 3 mutation vectors. {t_pop} given." 
        
        base = np.arange(0, t_pop, 1)
        m1 = np.random.permutation(t_pop)
        while np.any(m1 == base):
            m1 = np.random.permutation(t_pop)
            
        m2 = np.random.permutation(t_pop)
        while np.any(m2 == m1) or np.any(m2 == base):
            m2= np.random.permutation(t_pop)
            
        m3 = np.random.permutation(t_pop)
        while np.any(m3 == m1) or np.any(m3 == m2) or np.any(m3 == base):
            m3 = np.random.permutation(t_pop)
            
        return m1, m2, m3
    
    def setupLearner(
            self, children: Population):
        """
        Generate lists containing the bodies and brains of the population.
        
        :param children: Population of children.
        """
        
        bodies = [body.genotype.develop().body for body in children.individuals]
        brains = []
        sol_sizes = []
        
        for body in bodies:
            active_hinges = body.find_modules_of_type(ActiveHinge)
            brain = (
                cpg_network_structure,
                output_mapping,
            ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
            brains.append(brain)
            sol_sizes.append(cpg_network_structure.num_connections)
            
        return bodies, brains, sol_sizes
    
    def initialSolutions(
            self, population: Population) -> Population:
        """
        Generate random weights for the initial population.
        """

        _, _, sol_sizes = self.setupLearner(population)
        
        for idx, sol_size in enumerate(sol_sizes):
            if sol_size == 0: sol_size = 1 # Prevent empty list as output
            population.individuals[idx].solution = np.random.uniform(
                low=-1.0, high=1.0, size=sol_size).tolist()
            
        return population
    
    def setSolutionSizes(
            self, children: Population, sol_sizes = list[int]) -> Population:
        """
        Reformat solution vectors to the right sizes.

        :param children: Population of children.
        :param sol_sizes: Correct sizes of the solution vectors.
        """
        
        for idx, sol_size in enumerate(sol_sizes):
            if sol_size == 0: sol_size = 1 # Prevent empty list as output
            solution = children.individuals[idx].solution
            if len(solution) >= sol_size:
                solution = solution[:sol_size]
            else:
                sample = np.random.uniform(
                    low=-1.0, high=1.0, size = sol_size - len(solution))
                solution = np.concatenate((solution, sample)).tolist()
                
            children.individuals[idx].solution = solution
        
        return children

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
    
    def decodeSolutions(
            self, population: Population) -> Population:
        """
        Decode the solutions into the correct list format.
        
        :param population: Population of robots.
        """
        
        pass

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
        self, population: Population, children: Population
    ) -> tuple[Population, dict[str, Any]]:
        """
        Select survivors using a tournament.

        :param population: The population the parents come from.
        :param children: Population of children.
        :param kwargs: The offspring, with key 'offspring_population'.
        :returns: A newly created population.
        :raises ValueError: If the population is empty.
        """
        
        offspring, offspring_fitness, offspring_solution = self.setupChildren(children)

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
                        solution=population.individuals[i].solution,
                    )
                    for i in original_survivors
                ]
                + [
                    Individual(
                        genotype=offspring[i],
                        fitness=offspring_fitness[i],
                        solution=offspring_solution[i],
                    )
                    for i in offspring_survivors
                ]
            ),
            {},
        )
    
    def setupChildren(
            self, children: Population):
        """
        Extract the genotypes and fitnesses for correct formatting.
        
        :param children: Population of children.
        """
        
        genotypes = []
        fitnesses = []
        solutions = []
        
        for child in children.individuals:
            genotypes.append(child.genotype)
            fitnesses.append(child.fitness)
            solutions.append(child.solution)
            
        return genotypes, fitnesses, solutions
    
    def encodeSolutions(
            self, population: Population) -> str:
        """
        Convert solutions (list of floats) into a string for compatibility.
        
        :param population: Population of robots.
        """
        
        for i in range(len(population.individuals)):
            population.individuals[i].solution = json.dumps(
                population.individuals[i].solution)
            
        return population

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

        :param rng: The random generator.
        :param innov_db_body: The innovation database for the body.
        :param innov_db_brain: The innovation database for the brain.
        """
        self.rng = rng
        self.innov_db_body = innov_db_body
        self.innov_db_brain = innov_db_brain

    def reproduce(
        self, parentPairs: list[list[int]], 
        parent_population: dict()) -> list[Genotype]:
        """
        Reproduce the population by crossover.

        :param parentPairs: Pairs of parents for the children.
        :param population: Population of parents.
        :return: The genotypes of the children.
        """

        # Extract population and perform crossover/mutation
        parent_population = parent_population.get("parent_population")
        offspring_genotypes = [
            Genotype.crossover(
                parent_population.individuals[parent1_i].genotype,
                parent_population.individuals[parent2_i].genotype,
                self.rng,
            ).mutate(self.innov_db_body, self.innov_db_brain, self.rng)
            for parent1_i, parent2_i in parentPairs
        ]
    
        # Output population of children (no fitnesses/solutions yet)
        children = Population(
            individuals = [
                Individual(genotype = g_child,
                           fitness = None,
                           solution = None)
                for g_child in offspring_genotypes
                ]
            )
        
        children = self.insertSolution(children, parent_population, parentPairs)
        
        return children
    
    def insertSolution(
            self, children: Population, population: Population,
            parentPairs: list[list[int]]) -> Population:
        """
        Fill in the best performing parent's solution
        
        :param population: Population of all individuals minus the children.
        :param children: Population of children.
        """
        
        solutions = self.findParentSolutions(parentPairs, population)
        
        for idx, sol in enumerate(solutions):
            children.individuals[idx].solution = sol
        
        return children
        
    def findParentSolutions(
            self, parentPairs: list[list[int]], population: Population) -> list[list[float]]:
        """
        Finds the best parent solution for a child given its parent pair.

        :param parentPairs: List of parent index pairs.
        :param population: Population of parents.
        """
        best_solutions = []
        for (p1, p2) in parentPairs:
            if population.individuals[p1].fitness > population.individuals[p2].fitness:
                idx = p1
            else: idx = p2
            
            best_solutions.append(population.individuals[idx].solution)
            
        return(best_solutions)
    
        
        children = self.insertSolution(children, population, parentPairs)
        return(children)

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

# Create and save the experiment instance.
experiment = Experiment(rng_seed=rng_seed)
logging.debug("Saving experiment configuration.")
with Session(dbengine) as session:
    session.add(experiment)
    session.commit()
    
innov_db_body = multineat.InnovationDatabase()
innov_db_brain = multineat.InnovationDatabase()

learner = BrainOptimizerDE()    
parent_selector = ParentSelector(offspring_size=config.OFFSPRING_SIZE, rng=rng)
survivor_selector = SurvivorSelector(rng=rng)
crossover_reproducer = CrossoverReproducer(
    rng=rng, innov_db_body=innov_db_body, innov_db_brain=innov_db_brain
)

modular_robot_evolution = ModularRobotEvolution(
    parent_selection=parent_selector,
    survivor_selection=survivor_selector,
    reproducer=crossover_reproducer,
    learner=learner
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
        Individual(genotype=genotype, fitness_total=0.0, fitness_forward=0.0,
                   fitness_rot_l=0.0, fitness_rot_r=0.0, beta = 0.0
                   solution_forward=[], solution_rot_l=[], solution_rot_r=[]
                   )
        for genotype in initial_genotypes
        ]
    )

# Finish the zeroth generation and save it to the database.
generation = Generation(
    experiment=experiment, generation_index=0, population=population,
)
save_to_db(dbengine, generation)




