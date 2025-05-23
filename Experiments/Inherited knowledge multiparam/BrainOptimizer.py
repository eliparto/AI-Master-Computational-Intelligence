""" Brain Optimizer (Differential Evolution) """

import config
import numpy as np
import numpy.typing as npt
from typing import Any
from tqdm import tqdm
from database_components import (
    Base,
    Experiment,
    Generation,
    Genotype,
    Individual,
    Population,
)
from evaluator_brain_targeted_locomotion import Evaluator

from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.brain.cpg import CpgNetworkStructure
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)
from revolve2.experimentation.evolution.abstract_elements import Learner

class BrainOptimizerDE(Learner):
    """Optimizer class (Differential Evolution)"""
    
    def __init__(self, bounds) -> None:
        self.bounds = bounds
        
    def learn(
            self, population: Population, **kwargs: Any,) -> Population:
        """
        Generate individual robots from the population and optimize their weights
        for a targeted locomotion task.
        
        :param population: Population to go through DE.
        """
        # Generate children bodies and brains
        bodies, brains, solution_sizes = self.setupLearner(population)
        # Reformat solution vectors to the correct sizes
        population = self.setSolutionSizes(population, solution_sizes)
        # Generate targets to train a generation
        targets = self.generateTargets()
        
        for idx, body in enumerate(tqdm(bodies, leave = False, position = 0)):
            # Setup optimizer
            cpg_network_structure, output_mapping = brains[idx]
            
            # Only optimize robots with at least 2 joints
            if cpg_network_structure.num_connections > 0:
                solutions = population.individuals[idx].solutions
                nose = population.individuals[idx].nose
                assert nose >= 0, "No nose orientation. Call morpho.findNose() on population."
        
                evaluator = Evaluator(
                headless=True,
                num_simulators=config.NUM_SIMULATORS_BRAIN,
                cpg_network_structure=cpg_network_structure,
                body=body,
                output_mapping=output_mapping,
                nose=nose,
                targets=targets.copy()
                )
                
                sol_t, sol_c = self.generate_T_C(solutions)
                for gen in tqdm(range(config.NUM_GENERATIONS_BRAIN),
                                leave = False):
                    sol_next_gen, max_fit = self.optimize(sol_t, sol_c, evaluator)
                    sol_t, sol_c = self.generate_T_C(sol_next_gen)
                    
                # Update fitness and solutions
                population.individuals[idx].solutions = sol_next_gen[0].flatten('C').tolist()
                population.individuals[idx].fitness = max_fit
 
            # TODO: Do something when no. of hinges is not enough to optimize
            else:
                population.individuals[idx].fitness = -1000.0
                
        return population
    
    def generateTargets(self) -> npt.NDArray[np.float_]:
        """
        Generate list of target coordinates for robots to navigate to.
        """
        targets = np.random.randint(
            low=self.bounds[0], high=self.bounds[1], size=(20,2)
            ).astype(float)
        
        return targets
    
    def generate_T_C(
            self, T) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
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
        T = np.array(T)
        assert T.ndim == 1 or 2, f"Incorrect target matrix shape: {T.shape}"
        if T.ndim == 1: # One vector passed: expand into matrix and create pop of perturbed matrices
            T = np.stack([
                np.reshape(T, (3,int(len(T)/3)))
                ]*config.NUM_POPULATION_BRAIN
                )
            P_pop = np.random.normal(loc=0.0, scale=0.05, size=T.shape) # Perturbation
            T += P_pop
        elif T.ndim == 2: # List of vectors passed: turn each weight vector into weight matrix
            T = np.reshape(T, (config.NUM_POPULATION_BRAIN,3,int(len(T[0])/3)))
        
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
            evaluator) -> tuple[npt.NDArray[np.float_], float]:
        """
        Compare target vectors with candidate vectors for the next generation.
    
        :param T: Target vectors.
        :param C: Candidate solutions.
        """
        # Reshape matrices into solution vectors
        T = np.reshape(T, (len(T), len(T[0][0])*3))
        C = np.reshape(C, (len(C), len(C[0][0])*3))
        assert T.ndim == 2, f"Incorrect target matrix shape: {T.shape}"
        
        # Evaluate targets
        solutions = np.vstack((T, C))
        fitnesses = evaluator.evaluate(solutions)
        
        # Sort targets and betas by fitness indices (high to low)
        sort_idx = np.flip(np.argsort(fitnesses))
        solutions = solutions[sort_idx]
        
        return solutions[:config.NUM_POPULATION_BRAIN], max(fitnesses)
    
    def mutationIndices(
            self, t_pop) -> tuple(npt.NDArray[np.float_]):
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
            self, children: Population
            ) -> tuple[list[Body], # Bodies
                       list[tuple[CpgNetworkStructure, list[tuple[int, ActiveHinge]]]], # Brains
                       list[int]]: # Solution sizes
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
                low=-1.0, high=1.0, size=sol_size*3).tolist()
            
        return population
    
    def setSolutionSizes(
            self, children: Population, sol_sizes: list[int]) -> Population:
        """
        Reformat solution vectors to the right sizes.
        This is done by either concatenating weights to the right dimension
        or 'cutting off' unnecessary weights.

        :param children: Population of children.
        :param sol_sizes: Correct sizes of the solution vectors.
        """
        
        for idx, sol_size in enumerate(sol_sizes):
            if sol_size == 0: continue # Robots with no connections are skipped in learn()
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
    
    def _dummyFitnesses(
            self, population: Population) -> Population:
        """
        Generate dummy fitness values for a population.
        Only for testing.
        """
        for p in population.individuals:
            p.fitness = np.random.normal()
            
        return population
