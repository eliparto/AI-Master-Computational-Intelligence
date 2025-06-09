""" Analyze population diversity and idnividual novelty """

from BodyCheck import BodyCheck
from collections import defaultdict
import config_template as config
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from database_components import Experiment, Generation, Individual, Population, Genotype
from sqlalchemy import select
from sqlalchemy.orm import Session
from revolve2.experimentation.database import OpenMethod, open_database_sqlite

# Import database
dbName = "Databases/exp_inherit_2.sqlite"
dbengine = open_database_sqlite(
    dbName, open_method=OpenMethod.OPEN_IF_EXISTS
)

def calcDiversity(genotypes: list[Genotype]) -> list[float]:
    """
    Calculate the diversity of a generation of robots.
    First, a matrix of feature vectors is made, after which it is normalized
    per feature. Finally, we perform PCA to extract the ellipsoid volume as a
    measure of diversity.
    :param gen: List of all genotypes within a generation (size = pop.size).
    """
    bodies = [g.develop().body for g in genotypes]
    noses = morpho.findNose(bodies)
    
    # Construct matrix F of morpho feature vectors f
    F = [
            morpho.xyz_symmetry(bodies[idx]) + \
            morpho.count_bricks_hinges(bodies[idx]) + \
            morpho.calc_size_volume(bodies[idx]) + \
            morpho.findLimbs(bodies[idx]) + \
            [noses[idx]] for idx in range(len(bodies))
        ]
    
    # Normalize features (per column)
    scaler = StandardScaler()
    scaler.fit(F)
    F_scaled = scaler.transform(F)
    
    # Perform PCA decomposition
    pca = PCA()
    F_pca = pca.fit_transform(F_scaled)
    ellips_volume = np.prod(pca.explained_variance_)
        
    return F, F_scaled, ellips_volume
    
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

# Convert the list of rows per generation to list of genotypes
for experiment_id, generations in rows_by_experiment_and_generation.items():
    for generation_index, row_list in generations.items():
        genotypes = [row[2] for row in row_list]
        rows_by_experiment_and_generation[experiment_id][generation_index] = genotypes

experiments = [
        [
        rows_by_experiment_and_generation[i][j] for j in range(config.NUM_GENERATIONS_BODY+1)
        ] for i in range(1, len(rows_by_experiment_and_generation)+1)
    ]


    
morpho = BodyCheck()
