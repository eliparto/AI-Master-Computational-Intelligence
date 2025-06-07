""" Analyze population diversity and idnividual novelty """

from BodyCheck import BodyCheck
from collections import defaultdict
import config_template as config
import numpy as np

from database_components import Experiment, Generation, Individual, Population, Genotype
from sqlalchemy import select
from sqlalchemy.orm import Session
from revolve2.experimentation.database import OpenMethod, open_database_sqlite

# Import database
dbName = "Databases/newstate_waypoints_short_1.sqlite"
dbengine = open_database_sqlite(
    dbName, open_method=OpenMethod.OPEN_IF_EXISTS
)

def calcDiversity(genotypes: list[Genotype]) -> list[float]:
    """
    Calculate the diversity of a generation of robots.
    :param gen: List of all genotypes within a generation (size = pop.size).
    """
    bodies = [g.develop().body for g in genotypes]
    
    return None
    
#TODO: Expand to import JSON
with Session(dbengine) as ses:
    rows = ses.execute(
        select(
            Experiment.id.label("experiment_id"),
            Generation.generation_index,
            Genotype,
            Individual.fitness,
            Individual.solutions,
            Individual.nose
        )
        .join_from(Genotype, Individual, Genotype.id == Individual.genotype_id)
        .join_from(Individual, Population, Individual.population_id == Population.id)
        .join_from(Population, Generation, Population.id == Generation.population_id)
        .join_from(Generation, Experiment, Generation.experiment_id == Experiment.id)
        .order_by(Experiment.id, Generation.generation_index, Individual.fitness.asc())
    ).all()
    
rows_by_experiment_and_generation = defaultdict(lambda: defaultdict(list))

for row in rows:
    rows_by_experiment_and_generation[row.experiment_id][row.generation_index].append(row)

# Now convert the list of rows per generation to list of tuples
for experiment_id, generations in rows_by_experiment_and_generation.items():
    for generation_index, row_list in generations.items():
        # Extract tuples (Genotype, fitness, nose) for each row in this generation
        tuples_list = [(row[2], row[3], row[5]) for row in row_list]
        
        # Replace the list of rows with the list of tuples
        rows_by_experiment_and_generation[experiment_id][generation_index] = tuples_list
        
diversities = []
for i in range(len(rows_by_experiment_and_generation)):
    gen_genotypes = [
        rows_by_experiment_and_generation[i][j] for j in range(config.NUM_GENERATIONS_BODY+1)
        ]
    

morpho = BodyCheck()
