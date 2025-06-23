"""Rerun the n best robot between all combined experiments."""

import config
import json
from database_components import Genotype, Individual, Generation, Experiment, Population
from evaluator_brain_targeted_locomotion import Evaluator
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.engine.row import Row
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import pandas as pd
from BodyCheck import BodyCheck
from sklearn.neighbors import NearestNeighbors

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)

def extract(dbName: str) -> list[Row]:
    """
    Extract all experiments in a database file.
    """
    # Open database
    dbengine = open_database_sqlite(
        dbName, open_method=OpenMethod.OPEN_IF_EXISTS
    )
    
    # Extract experiments
    with Session(dbengine) as ses:
        rows = ses.execute(
            select(
                Experiment.id.label("experiment_id"),
                Generation.generation_index,
                Genotype,
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
            rows_by_experiment_and_generation[experiment_id][generation_index] = (
                genotypes
                )

    experiments = [
            [
            rows_by_experiment_and_generation[i][j] for j in range(config.NUM_GENERATIONS_BODY+1)
            ] for i in range(1, len(rows_by_experiment_and_generation)+1)
        ]
    
    return experiments

def morphVect(gens: list[list[Genotype]]) -> npt.NDArray[np.float_]:
    """
    Generate morphological feature vectors.
    :param gen: list of population genotypes per generation.
    """
    morpho = BodyCheck()
    print("Generating morpho feature vectors..")
    
    allFeatures = []
    for gen in tqdm(gens, leave = True):
        bodies = [g.develop().body for g in gen]
        xyz_sym = [morpho.xyz_symmetry(body) for body in bodies]
        noses = np.array(morpho.findNose(bodies))
        
        # Flip x and y symmetries based on nose orientation
        for i, nose in enumerate(noses):
            if nose % 2 == 0:
                temp = xyz_sym[i]
                xyz_sym[i][0] = temp[1]
                xyz_sym[i][1] = temp[0]
        xyz_sym = np.array(xyz_sym)
        noses = np.reshape(n, (len(n),1))
        
        # Generate morphological feature vectors
        features = np.array([
                morpho.count_bricks_hinges(body) + \
                morpho.calc_size_volume(body) + \
                morpho.findLimbs(body) for body in bodies
            ])
        features = np.hstack((features, noses))
        features = np.hstack((features, xyz_sym))
        
        # Normalize feature vectors columnwise and append
        features = features / features.max(axis=0)
        allFeatures.append(features)
        
    return np.array(allFeatures)

def KNNdiversity(gen: list[Genotype]) -> pd.DataFrame:
    """
    Generate a dataframe containing the diversity per generation of an experiment.
    :param gen: list of population genotypes per generation.
    """
    ...

def main() -> None:
    """Perform the rerun."""
    # Check for passed arguments
    parser = argparse.ArgumentParser(description="Plot the n best trajectories for a test condition.")
    parser.add_argument("-name", type=str, help="Specify the database JSON filename.")
    parser.add_argument("-figName", type=str, default = None, help="Specify figure title.")
    args = parser.parse_args()
        
    if args.name:
        assert args.t > 0, "Invalid simulation time passed."
        
        # Import JSON file containing database names
        dbList = args.name + ".json"
        with open(dbList, 'r') as file:
            dbFile = json.load(file)    
        dbs = dbFile.get("Databases", [])
        assert len(dbs) > 0, "Need to import at least 2 databases to combine."
        
        # Extract data from all experiments
        allExps = []
        for db in dbs:
            dbName = "Databases/" + db + ".sqlite"
            exps = extract(dbName)
            allExps += exps
        
    else: print("Pass database name with '-name'. Closing now.")

if __name__ == "__main__":
    main()
