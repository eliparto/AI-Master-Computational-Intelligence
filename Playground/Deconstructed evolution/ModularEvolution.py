""" Evolutionary Processes """

from typing import Any
import config
import multineat
import numpy as np
import numpy.typing as npt
from database_components import (
    Base,
    Experiment,
    Generation,
    Genotype,
    Individual,
    Population,
)
from revolve2.experimentation.evolution.abstract_elements import Reproducer, Selector
from revolve2.experimentation.optimization.ea import population_management, selection

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
