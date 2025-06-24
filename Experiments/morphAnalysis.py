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
from typing import Union

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)
# Manually adjust font size globally
plt.rcParams.update({
    "font.size": 10,         # Default text size
    "axes.titlesize": 35,    # Title font size
    "axes.labelsize": 35,    # Axis label size
    "xtick.labelsize": 30,   # Tick label size
    "ytick.labelsize": 30,
    "legend.fontsize": 30,
    "figure.titlesize": 35,
    "figure.figsize": (15,12)
})

k = 7 # Floor of square root of pop size

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
        noses = np.reshape(noses, (len(noses),1))
        
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

def KNNdiversity(exp: Union[list[list[Genotype]], None],
                 k: int, vectors: Union[npt.NDArray[np.float_], None] = None, 
                 generate: bool = True) -> float:
    """
    Determine a generations diversity metric.
    Diversity is expressed by the average distance to the k nearest neighbors
    for all individuals in a population.
    :param gen: list of population genotypes per generation.
    :param k: k-value (k+1 used as self used as first neighbor)
    :param generate: Set True to generate morphological feature vectors.
    :param vectors: (Optional) array of morhpological feature vectors to bypass vector generation.
    """
    # Generate tensor of morpho feature matrices per generation if prompted
    if generate: gens = morphVect(exp)
    else: gens = vectors
    
    # Calculate diversity using kNN distances per generation
    divs = []
    for gen in gens:
        knn = NearestNeighbors(
            n_neighbors=k+1,
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

def optimizeK(
        k_vals: list[int], f_tensor = npt.NDArray[np.float_]
        ) -> list[float]:
    """
    Simple 1-D grid search optimizer for the k-value.
    :param exps: List of all experiments within a database.
    :param k_vals: List of k-values to try.
    :param vectors: Tensor containing enerated morphological feature vectors.
    """
    k_results = []
    for k in k_vals:
        diversities = []
        for f_matrix in f_tensor: # ~> for exp in exps
            diversities.append(
                KNNdiversity(
                    exp=None, k=k, vectors=f_matrix, 
                    generate=False))
            
        diversities = np.array(diversities)
        div_avg = np.average(diversities, axis=0)
        div_std = np.std(diversities, axis = 0)
        
        k_results.append([div_avg, div_std])
                        
    return k_results
    
def calcDiversity(
        exps: list[list[list[Genotype]]], k: int,
        ) -> npt.NDArray[np.float_]:
    """
    Calculate diversities over all experiments.
    """
    diversities = []
    for exp in tqdm(exps, leave=True, position=0):
        diversities.append(KNNdiversity(exp=exp, k=k))
        
    diversities = np.array(diversities)
    div_avg = np.average(diversities, axis=0)
    div_std = np.std(diversities, axis = 0)
    
    return div_avg, div_std

def plotDiv(
        avg_1: npt.NDArray[np.float_], std_1: npt.NDArray[np.float_],
        figName: Union[str, None],
        ) -> None:
    """
    Plot the mean and std shaded diversities for a single experiment.
    """
    plt.figure()
    plt.plot(np.arange(0,config.NUM_GENERATIONS_BODY+1,1), avg_1, 
             c="pink", lw=4)
    plt.fill_between(np.arange(0,config.NUM_GENERATIONS_BODY+1,1), 
                     avg_1-std_1, avg_1+std_1, color="pink", alpha=0.4)
    
    if figName != None: plt.title(figName)
    plt.grid()
    plt.xlabel("Generation index")
    plt.ylabel("Diversity")
    plt.xlim([0,config.NUM_GENERATIONS_BODY])
    plt.ylim([0,1])
    plt.show()

def plotDivCompare(
        avg_1: npt.NDArray[np.float_], std_1: npt.NDArray[np.float_],
        avg_2: npt.NDArray[np.float_], std_2: npt.NDArray[np.float_],
        figName: Union[str, None] = None,
        ) -> None:
    """
    Plot the mean and std shaded diversities for two experiments.
    Pass warm start vars first.
    """
    plt.figure()
    plt.plot(np.arange(0,config.NUM_GENERATIONS_BODY+1,1), avg_2, 
             c="deeppink", lw=4, label="Warm start")
    plt.fill_between(np.arange(0,config.NUM_GENERATIONS_BODY+1,1), 
                     avg_2-std_2, avg_2+std_2, color="deeppink", alpha=0.25)
    plt.plot(np.arange(0,config.NUM_GENERATIONS_BODY+1,1), avg_1, 
             c="lightseagreen", lw=4, label="Cold start")
    plt.fill_between(np.arange(0,config.NUM_GENERATIONS_BODY+1,1), 
                     avg_1-std_1, avg_1+std_1, color="lightseagreen", alpha=0.25)
    
    plt.grid()
    title = "Morphological diversity accross repetitions with std as shade\n"
    if figName != None: title += figName
    plt.title(title)
    plt.xlabel("Generation index")
    plt.ylabel("Diversity")
    plt.xlim([0,config.NUM_GENERATIONS_BODY])
    plt.ylim([0.3,0.9])
    plt.legend()
    plt.show()
    
def outputCSV() -> None:
    ...

def main() -> None:
    """Perform the rerun."""
    # Check for passed arguments
    parser = argparse.ArgumentParser(description="Plot the n best trajectories for a test condition.")
    parser.add_argument("-db1", type=str, help="Specify the cold start JSON database filename.")
    parser.add_argument("-db2", type=str, help="Specify the warm start JSON database filename.")
    parser.add_argument("-figName", type=str, default = None, help="Specify figure title.")
    parser.add_argument("-div", action="store_true", help="Plot diversities")
    parser.add_argument("-nov", action="store_true", help="Plot novelties")
    args = parser.parse_args()
    
    # Import JSON file containing database names
    dbLists = [args.db1 + ".json", args.db2 + ".json"]
    conditions = []
    for dbList in dbLists:
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
            
        conditions.apppend(allExps)
        
    # Generate morphological feature vectors/tensors
    tensor_warm = np.array([
        morphVect(exp) for exp in conditions[0]
        ])
    tensor_cold = np.array([
        morphVect(exp) for exp in conditions[1]
        ])
    
    # Calculate and plot diversities
    if args.div:
        div_warm = []
        div_cold = []
        for f_matrix in tensor_warm:
            div_warm.append(KNNdiversity(exp=f_matrix, k=k))
        for f_matrix in tensor_cold:
            div_cold.append(KNNdiversity(exp=f_matrix, k=k))
        
        div_warm = np.array(div_warm)
        div_cold = np.array(div_cold)
        div_warm_avg = np.average(div_warm, axis=0)
        div_warm_std = np.std(div_warm, axis = 0)
        div_cold_avg = np.average(div_cold, axis=0)
        div_cold_std = np.std(div_cold, axis = 0)
        
        plotDivCompare(div_warm_avg, div_warm_std, div_cold_avg, div_cold_std)
    
    if args.nov:
        ...

if __name__ == "__main__":
    main()
