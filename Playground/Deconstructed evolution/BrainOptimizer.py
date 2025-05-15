""" Brain Optimizer (Differential Evolution) """

import config
import logging
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
from evaluator_brain_script import Evaluator as Evaluator_brain
from evaluator_brain_testing import Evaluator as Evaluator_beta

from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)
from revolve2.experimentation.evolution.abstract_elements import Learner


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
            
            # Only optimize robots with at least 2 joints
            if cpg_network_structure.num_connections > 0:
                evaluator = Evaluator_beta(
                headless=True,
                num_simulators=config.NUM_SIMULATORS_BRAIN,
                cpg_network_structure=cpg_network_structure,
                body=body,
                output_mapping=output_mapping,
                targets=targets,
                )
                
                solutions = population.individuals[idx].solutions
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
                population.individuals[idx].fitness = -1000
                
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
            fit_type: int, evaluator) -> tuple[list[float], float, float]:
        """
        Compare target vectors with candidate vectors for the next generation.
    
        :param T: Target vectors.
        :param C: Candidate solutions.
        """
        # TODO: Remove 'betas'
        
        logging.debug("DE: Comparing targets with candidates")
        # Evaluate targets
        solutions = np.vstack((T, C))
        fitnesses, betas = evaluator.evaluate(solutions, fit_type)
        
        # Sort targets and betas by fitness indices (high to low)
        sort_idx = np.flip(np.argsort(fitnesses))
        solutions = solutions[sort_idx]
        betas = betas[sort_idx]
        
        return solutions[:config.NUM_POPULATION_BRAIN], max(fitnesses), betas[0]
    
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
