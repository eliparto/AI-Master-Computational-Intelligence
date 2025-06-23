"""Calculate diversity and novelty metrics over populations."""

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
from matplotlib import pyplot as plt

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

def morphVect(exp: list[list[Genotype]]) -> npt.NDArray[np.float_]:
    """
    Generate morphological feature vectors.
    :param exp: An experiment -> list containing genotypes for all its generations.
    """
    morpho = BodyCheck()
    allFeatures = []
    for gen in tqdm(exp, leave=False, position=1):
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

def KNNdiversity(exp: list[list[Genotype]]) -> float:
    """
    Determine a generations diversity metric.
    Diversity is expressed by the average distance to the k nearest neighbors
    for all individuals in a population.
    :param gen: list of population genotypes per generation.
    """
    # Generate tensor of morpho feature matrices per generation
    gens = morphVect(exp)
    
    # Calculate diversity using kNN distances per generation
    divs = []
    for gen in gens:
        knn = NearestNeighbors(
            n_neighbors=int(np.floor(np.sqrt(config.POPULATION_SIZE_BODY)))+1,
            algorithm="auto",
            metric="euclidean"
            )
        
        knn.fit(gen)
        dist, _ = knn.kneighbors(gen, return_distance=True)
        avg_dist = np.average(dist[:,1:], axis=1) # Remove distances to self
        
        # Diversity: Average distance of average distances
        div = np.average(avg_dist)
        divs.append(div)
        
    return np.array(divs)

def calcDiversity(exps: list[list[list[Genotype]]]) -> npt.NDArray[np.float_]:
    """
    Calculate diversities over all experiments.
    """
    diversities = []
    for exp in tqdm(exps, leave=True, position=0):
        diversities.append(KNNdiversity(exp))
        
    diversities = np.array(diversities)
    div_avg = np.average(diversities, axis=0)
    div_std = np.std(diversities, axis = 0)
    
    return div_avg, div_std

def plotDiv(
        avg_1: npt.NDArray[np.float_], std_1: npt.NDArray[np.float_],
        avg_2: npt.NDArray[np.float_], std_2: npt.NDArray[np.float_]
        ) -> None:
    """
    Plot the mean and std shaded diversities.
    """
    plt.plot(np.arange(0,config.NUM_GENERATIONS_BODY+1,1), avg_1, 
             c="lightblue", lw=4)
    plt.fill_between(np.arange(0,config.NUM_GENERATIONS_BODY+1,1), 
                     avg_1-std_1, avg_1+std_1, color="lightblue", alpha=0.4)
    
    plt.plot(np.arange(0,config.NUM_GENERATIONS_BODY+1,1), avg_2, 
             c="orange", lw=4)
    plt.fill_between(np.arange(0,config.NUM_GENERATIONS_BODY+1,1), 
                     avg_2-std_2, avg_2+std_2, color="orange", alpha=0.25)
    
    plt.grid()
    plt.xlabel("Generation index")
    plt.ylabel("Diversity")
    plt.xlim([0,config.NUM_GENERATIONS_BODY+1])
    plt.ylim([0,1])

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
