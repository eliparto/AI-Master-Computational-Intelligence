""" Deconsturcted generational step for variable exploration 
TODO:   - Remove unused functions (encode/decode)
        - Remove 'parent_pop' from crossoverReproducer
        - Plot/report fitnesses after steps to confirm learning
        - Go over all type castings
        - Investigate how to determine total fitness -> sum/(weighted) avg etc.
        - Remove usage of 'children' -> confusion with 'population'
        - Check whether xy displacement and orientations are correct
        - Put modules in their respective folders
        - Targeted locomotion
"""

import logging
import os
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
from evaluator_brain_script import Evaluator as Evaluator_brain
from evaluator_brain_testing import Evaluator as Evaluator_beta
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body import Module

from revolve2.modular_robot.body.base import ActiveHinge, Brick, Core, Body
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.evolution import ModularRobotEvolution
from revolve2.experimentation.evolution.abstract_elements import Reproducer, Selector, Learner
from revolve2.experimentation.optimization.ea import population_management, selection
from revolve2.experimentation.rng import make_rng, seed_from_time

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic, BrainCpgNetworkLocomotion, CpgNetworkStructure
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

from BodyCheck import BodyCheck
# TODO: Check compatability
#from BrainOptimizer import BrainOptimizerDE
#from ModularEvolution import ParentSelector, CrossoverReproducer, SurvivorSelector

# Brain optimizer
class BrainOptimizerDE(Learner):
    """Optimizer class (DE)"""
    
    def __init__(self) -> None:
        self
        
    def learn(
            self, population: Population,  
            targets = list[list[float]], **kwargs: Any,) -> Population:
        """
        Generate individual robots from the population and optimize their weights.
        This process optimizes weights for 1). XY displacement; 2/3). Rotating left/right.
        In the movement optimization loop, the first iteration (i == 0) also updates beta.
        
        :param population: Population to go through DE.
        """
        
        # Generate children bodies and brains
        bodies, brains, solution_sizes = self.setupLearner(population)
        
        # Reformat solution vectors to the correct sizes
        population = self.setSolutionSizes(population, solution_sizes)
        
        print("Body:")
        for idx, body in enumerate(tqdm(bodies, leave = False, position = 0)):
            # Setup optimizer
            cpg_network_structure, output_mapping = brains[idx]
            targets = [[3,4]]
            
            # Only optimize robots with at least 2 joints
            if cpg_network_structure.num_connections > 0:
                solutions = population.individuals[idx].solutions
                nose = population.individuals[idx].nose
                
                evaluator = Evaluator_beta(
                headless=True,
                num_simulators=config.NUM_SIMULATORS_BRAIN,
                cpg_network_structure=cpg_network_structure,
                body=body,
                output_mapping=output_mapping,
                targets=targets,
                nose=nose,
                )
                
                sol_t, sol_c = self.generate_T_C(solutions)
                
                for gen in tqdm(range(config.NUM_GENERATIONS_BRAIN),
                                leave = False):
                    targets, max_fit, _ = self.optimize(sol_t, sol_c, evaluator)
                    sol_t, sol_c = self.generate_T_C(targets)
                    
                # Update fitness and solution
                population.individuals[idx].solutions = targets[0].flatten('C').tolist()
                population.individuals[idx].fitness = max_fit
 
            # TODO: De something when no. of hinges is not enough to optimize
            else:
                population.individuals[idx].fitness = -1000.0
                
        return population
    
    def generate_T_C(
            self, T):
        """
        Generates target and candidate vectors for Differential Evolution).
        
        :param vectors: Cadidate solution(s) to go through DE. Can be 2D matrix or 3D tensor.
        
        T ->    Target vectors (can also be initial solution):
                Add perturbation vectors P to copies of the input vector.
                T = T + P w/ P ~ N(o, sd)
        M ->    Mutation vectors:
                m_i = t_a + F(t_b - t_c) w/ a, b, and c some random indices.
        C ->    Crossover vectors:
                Every m_i gets a binary crossover mask with prob_cr to mix between m_i and t_i.
        C is outputted to be compared to T. The winning genes get passed on.
        """
        # Create slightly perturbed population tensor of target matrices for the initial solution
        T = np.array(T)
        if T.ndim == 1:
            T = np.stack([
                np.reshape(T, (3,int(len(T)/3)))
                ]*config.NUM_POPULATION_BRAIN
                )
            P_pop = np.random.normal(loc=0.0, scale=0.05, size=T.shape)
            T += P_pop
            
        # Create tensor of perturbation matrices
        m_1, m_2, m_3 = self.mutationIndices(len(T))
        M = T[m_1] + config.F * (T[m_2] - T[m_3])
           
        # Crossover (use binary mask to decide if T or C is used) and clip
        cr_mask = np.random.choice(
            [0,1], size = T.shape, p = [1 - config.P_CR, config.P_CR]
            )
        C = np.where(cr_mask == 1, M, T)
        C = np.clip(C, a_min=-1.0, a_max=1.0)
        T = np.clip(T, a_min=-1.0, a_max=1.0)
        
        return T, C
    
    def optimize(
            self, T: npt.NDArray[np.float_], C: npt.NDArray[np.float_],
            evaluator) -> tuple[list[float], float, float]:
        """
        Compare target vectors with candidate vectors for the next generation.
    
        :param T: Target vectors.
        :param C: Candidate solutions.
        """
        # TODO: Remove 'betas'
        
        logging.debug("DE: Comparing targets with candidates")
        # Evaluate targets
        solutions = np.vstack((T, C))
        fitnesses = evaluator.evaluate(solutions)
        
        # Sort targets and betas by fitness indices (high to low)
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
            population.individuals[idx].solutions = np.random.uniform(
                low=-1.0, high=1.0, size=sol_size*3)
            
        return population
    
    def setSolutionSizes(
            self, children: Population, sol_sizes: list[int]) -> Population:
        """
        Reformat solution vectors to the right sizes.

        :param children: Population of children.
        :param sol_sizes: Correct sizes of the solution vectors.
        """
        
        for idx, sol_size in enumerate(sol_sizes):
            solutions = children.individuals[idx].solutions
            solutions = np.reshape(solutions, (3, int(len(solutions)/3)))
            
            # If solutions are too long -> cut off unnecessary part
            if solutions.shape[1] >= sol_size:
                solutions = np.hsplit(
                    solutions, np.array([sol_size, solutions.shape[1] - sol_size])
                    )[0]
                
            # If too short -> Sample necessary weights and add
            else:
                samples = np.random.uniform(
                    low=-1.0, high=1.0, size=(3, sol_size-solutions.shape[1]))
                solutions = np.hstack((solutions, samples))
                
            children.individuals[idx].solutions = solutions.flatten('C').tolist()
        
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
        )

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
    ) -> Population:
        """
        Select survivors using a tournament.

        :param population: The population the parents come from.
        :param children: Population of children.
        :returns: A newly created population.
        :raises ValueError: If the population is empty.
        """
        
        # Retrieve information from children
        (
            offspring, off_fitness_vectors, off_fitnesses, 
            off_betas, off_solutions
        ) = self.setupChildren(children) # TODO: Calculate (weighted) average fitnesses
        
        original_survivors, offspring_survivors = population_management.steady_state(
            old_genotypes=[i.genotype for i in population.individuals],
            old_fitnesses=[i.fitness for i in population.individuals],
            new_genotypes=offspring,
            new_fitnesses=off_fitnesses,
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
                        fitnesses=population.individuals[i].fitnesses,
                        solutions=population.individuals[i].solutions,
                        beta=population.individuals[i].beta,
                    )
                    for i in original_survivors
                ]
                + [
                    Individual(
                        genotype=offspring[i],
                        fitness=off_fitnesses[i],
                        fitnesses=off_fitness_vectors[i],
                        solutions=off_solutions[i],
                        beta=off_betas[i],
                    )
                    for i in offspring_survivors
                ]
            )
        )
    
    def setupChildren(
            self, children: Population):
        """
        Extract the genotypes and fitnesses for correct formatting.
        
        :param children: Population of children.
        """
        
        genotypes = []
        fitness_values = []
        fitness_vectors = []
        solutions = []
        betas = []
        
        for child in children.individuals:
            genotypes.append(child.genotype)
            fitness_values.append(child.fitness)
            fitness_vectors.append(child.fitnesses)
            solutions.append(child.solutions)
            betas.append(child.beta)
            
        return (genotypes, fitness_vectors, fitness_values, betas, solutions)

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
        parent_population: Population) -> list[Genotype]:
        """
        Reproduce the population by crossover.

        :param parentPairs: Pairs of parents for the children.
        :param population: Population of parents.
        :return: The genotypes of the children.
        """

        # Extract population and perform crossover/mutation
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
                Individual(genotype=g_child, fitness=0.0, fitnesses=3*[0.0], 
                           beta = 0.0, solutions=[]
                           )
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
            children.individuals[idx].solutions = sol
            
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
            
            best_solutions.append([population.individuals[idx].solutions])
            
        return(best_solutions) 

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
        Individual(genotype=genotype, fitness=0.0, nose = -1, solutions=[]
                   )
        for genotype in initial_genotypes
        ]
    )

body_check = BodyCheck(population, learner.setupLearner)

# Dummy target for locomotion
targets = [[10.0, 5.0]]

# Finish the zeroth generation and save it to the database.
generation = Generation(
    experiment=experiment, generation_index=0, population=population,
)
save_to_db(dbengine, generation)





