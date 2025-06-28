"""Visualize and show metrics for the best performing robots."""

import config
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from typing import Union
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

from BodyCheck import BodyCheck
from database_components import Genotype, Individual, Generation, Experiment, Population
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.engine.row import Row
from revolve2.experimentation.database import OpenMethod, open_database_sqlite

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
morpho = BodyCheck()

def importDB(dbList: str) -> list[Row]:
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
        
    return allExps

def extract(dbName: str) -> list[Genotype]:
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
                Individual.fitness,
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
            fitnesses = [row[3] for row in row_list]
            rows_by_experiment_and_generation[experiment_id][generation_index] = (
                genotypes
                )

    experiments = [
            [
            rows_by_experiment_and_generation[i][j] for j in range(config.NUM_GENERATIONS_BODY+1)
            ] for i in range(1, len(rows_by_experiment_and_generation)+1)
        ]
    
    return experiments

def bestRobots(exps: list[list[tuple[Genotype, float]]]) -> list[Genotype]:
    """
    Extract every experiment's best robot from the last generation.
    """
    robots = [
        gen[-1] for gen in [
                exp[-1] for exp in exps
            ]
        ]
    
    return robots

def plotRobots(genotypes = list[Genotype], fitness = list[float]) -> None:
    """
    Show all best robots for a condition.
    """
    # Develop bodies from genotype
    robots = [
        genotype.develop().body for genotype in genotypes
        ]
    
    for robot in robots:
        morpho.show2D(robot)

def morphVect(gens: list[Genotype]) -> npt.NDArray[np.float_]:
    """
    Generate morphological feature vectors.
    :param gens: List of genotypes.
    """
    bodies = [g.develop().body for g in gens]
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
    #features = features / features.max(axis=0)
    mask = np.where(features < 1e-8, 0, 1)
    features = features * mask
        
    return np.array(features)

def exportDF(features: npt.NDArray[Union[np.float_, np.int_]]) -> pd.DataFrame:
    """
    Generate a dataframe of morphological features for statistical analysis.
    """
    col_names = [
        "cnt_bricks", "cnt_joints", "vol_bbox", "vol_disp", "cnt_limb", 
        "avg_len_limb", "nose", "sym_x", "sym_y","sym_z"]
    df = pd.DataFrame(features, columns = col_names)
    
    return df

def main() -> None:
    """Perform the rerun."""
    # Check for passed arguments
    parser = argparse.ArgumentParser(description="Plot the diversites and long- and short-term novelties of two experiments.")
    parser.add_argument("-db", action="store_true", help="Toggle to import databases via JSON")
    parser.add_argument("-db1", type=str, help="Specify the cold start JSON database filename.")
    parser.add_argument("-db2", type=str, help="Specify the warm start JSON database filename.")
    parser.add_argument("-figName", type=str, default=None, help="Specify figure title.")
    parser.add_argument("-p", action="store_true", help="Plot diversities and novelties")
    parser.add_argument("-df", action="store_true", help="Output diversity and novelty dataframes")
    parser.add_argument("-dfName", type=str, default=None, help="Specify dataframe name.")
    args = parser.parse_args()
    
    if args.db1 and args.db2:
    
        # Import JSON file containing database names
        if args.db:
            dbList_warm = args.db1 + ".json"
            dbList_cold = args.db2 + ".json"
            allExps_warm = importDB(dbList_warm)
            allExps_cold = importDB(dbList_cold)    
        
        # Collect last generation's best robots per experiment and condition
        genotypes_warm = bestRobots(allExps_warm)
        genotypes_cold = bestRobots(allExps_cold)
        
        # Show robots
        plotRobots(genotypes_warm)
        plotRobots(genotypes_cold)
        
        # Generate + export DataFrames
        df_warm = exportDF(morphVect(genotypes_warm))
        df_cold = exportDF(morphVect(genotypes_cold))
        # df_warm.to_csv(dfName_warm)
        # df_cold.to_csv(dfName_cold)
        
    else: print("Pass databases with -db1 and -db2. Closing.")

if __name__ == "__main__":
    main()
