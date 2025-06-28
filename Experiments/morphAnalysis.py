"""Calculate diversity and novelty metrics over populations."""

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

def KNNdiversity(exp: npt.NDArray[np.float_], k: int
                 ) -> npt.NDArray[np.float_]:
    """
    Determine a generation's diversity metric.
    Diversity is expressed by the average distance to the k nearest neighbors
    for all individuals in a population.
    :param gen: list of population genotypes per generation.
    :param k: k-value (k+1 used as self used as first neighbor)
    :param generate: Set True to generate morphological feature vectors.
    :param vectors: (Optional) array of morhpological feature vectors to bypass vector generation.
    """
    # Calculate diversity using kNN distances per generation
    divs = []
    for gen in exp:
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

def KNNnovelty_longterm(gens: npt.NDArray[np.float_], k: int,
                 ) -> npt.NDArray[np.float_]:
    """
    Determine a generation's historic novelty metric.
    Novelty is expressed as the average distance from a population at generation g
    to an archive of populations from generation 0 to generation g-1.
    """
    # Calculate novelty using archive of gen 0 to g-1
    novelties = []
    for gen_idx, gen in enumerate(gens[1:]):
        # Create archive of features till previous generation
         archive = np.concatenate(gens[0:gen_idx+1])
         # Calculate distances from current generation to archive
         all_dist = distance.cdist(gen, archive, metric="euclidean")
         # Sort and choose k nearest neighbors
         k = int(np.floor(np.sqrt(len(archive))))
         all_dist = np.sort(all_dist)[:,:k]
         # Average per individual, then average over population
         all_dist = np.average(all_dist, axis=1)
         novelty = np.average(all_dist)
         novelties.append(novelty)
         
    return np.array(novelties)
         
def KNNnovelty_shortterm(gens: npt.NDArray[np.float_], k:int
                 ) -> npt.NDArray[np.float_]:
    """
    Determine a generations short-term novelty compared to the previous generation.
    See long term implementation for steps taken.
    """
    novelties = []
    for i in range(1, len(gens)):
        # Define previous and current generation
        gen_prev = gens[i-1]
        gen_curr = gens[i]
        # Calculate distances from current to previous generation
        all_dist = distance.cdist(gen_curr, gen_prev, metric="euclidean")
        all_dist = np.sort(all_dist, axis=1)[:,:k]
        all_dist = np.average(all_dist, axis=0)
        novelty = np.average(all_dist)
        novelties.append(novelty)
        
    return np.array(novelties)

def calcNovelty(exps: npt.NDArray[np.float_], k:int, longterm=True, 
                df: bool=False) -> npt.NDArray[np.float_]:
    """
    Calculate novelties over all experiments.
    :param lognterm: Toggle between long-term and short-term novelty.
    :param df: Toggle to output novelties without averaging.
    """
    if longterm: knnNovelty = KNNnovelty_longterm
    else: knnNovelty = KNNnovelty_shortterm
    novelties = []
    for exp in exps:
        novelties.append(knnNovelty(exp, k))
        
    novelties = np.array(novelties)
    nov_avg = np.average(novelties, axis=0)
    nov_std = np.std(novelties, axis=0)
    
    if df: return novelties
    else: return nov_avg, nov_std   
     
def calcDiversity(exps: npt.NDArray[np.float_], k: int, df: bool=False,
                  ) -> npt.NDArray[np.float_]:
    """
    Calculate diversities over all experiments.
    :param df: Toggle to output diversities without averaging
    """
    diversities = []
    for exp in exps:
        diversities.append(KNNdiversity(exp=exp, k=k))
        
    diversities = np.array(diversities)
    div_avg = np.average(diversities, axis=0)
    div_std = np.std(diversities, axis = 0)
    
    if df: return diversities
    else: return div_avg, div_std

def plotSingle(
        avg: npt.NDArray[np.float_], std: npt.NDArray[np.float_],
        figName: Union[str, None],
        ) -> None:
    """
    Plot the mean and std shaded diversities for a single experiment.
    """
    plt.figure()
    plt.plot(np.arange(0,len(avg),1), avg, 
             c="pink", lw=4)
    plt.fill_between(np.arange(0,len(avg),1), 
                     avg-std, avg+std, color="pink", alpha=0.4)
    
    if figName != None: plt.title(figName)
    plt.grid()
    plt.xlabel("Generation index")
    plt.ylabel("Diversity")
    plt.xlim([0,len(avg)])
    plt.ylim([0,1])
    plt.show()

def plotCompare(
        avg_1: npt.NDArray[np.float_], std_1: npt.NDArray[np.float_],
        avg_2: npt.NDArray[np.float_], std_2: npt.NDArray[np.float_],
        figName: Union[str, None] = None, novelty: bool = False, 
        longterm: bool = False,
        ) -> None:
    """
    Plot the mean and std shaded diversities for two experiments.
    Pass warm start vars first.
    TODO: Alter for novelty plotting -> start generation index at 1 and custom colors
    :param avg_i: Array of average novelty or diversity over generations.
    :param std_i: Array of std of novelty or diversity over generations.
    :param figName: Optional addition to figure title.
    :param novelty: Toggle between novelty and diversity plotting (layout changes etc.).
    """

    lim_div = [0.35, 0.85]
    lim_nov_short = [0.3, 0.8]
    lim_nov_long = [0.15, 0.8]
    plt.figure()
    if novelty:
        c1= "salmon"
        c2 = "darkcyan"
        x = np.arange(1,len(avg_1)+1,1)
    else: 
        c1 = "deeppink"
        c2 = "lightseagreen"
        x = np.arange(0,len(avg_1),1)
    
    plt.figure()
    plt.plot(x, avg_1, 
             c=c1, lw=4, label="Warm start")
    plt.fill_between(x, avg_1-std_1, avg_1+std_1, 
                     color=c1, alpha=0.25)
    plt.plot(x, avg_2, 
             c=c2, lw=4, label="Cold start")
    plt.fill_between(x, avg_2-std_2, avg_2+std_2, 
                     color=c2, alpha=0.25)
    
    if novelty:
        if longterm: 
            title = "Long-term novelty across repetitions with std as shade\n"
            plt.ylim(lim_nov_long)
        else: 
            title = "Short-term novelty across repetitions with std as shade\n"
            plt.ylim(lim_nov_short)
        plt.ylabel("Novelty")
        plt.xlim([1,len(avg_1)])

    else: 
        title = "Morphological diversity across repetitions with std as shade\n"
        plt.ylabel("Diversity")
        plt.xlim([0,len(avg_1)-1])
        plt.ylim(lim_div)


    # Appearance
    plt.title(title)
    plt.grid()
    plt.xlabel("Generation index")
    plt.legend()
    plt.show()
    
def outputCSV() -> None:
    """
    Export the final generation's diversities and novelties in CSV format.
    """
    ...

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
        
            # Generate morphological feature vectors/tensors
            print("Generating warm condition vectors..")
            tensor_warm = np.array([
                morphVect(exp) for exp in tqdm(allExps_warm)
                ])
            print("Generating cold condition vectors..")
            tensor_cold = np.array([
                morphVect(exp) for exp in tqdm(allExps_cold)
                ])    
        
        else:
            tensor_warm = np.load(args.db1 + ".npy")
            tensor_cold = np.load(args.db2 + ".npy")
        
        if args.p: # Plot diversities and novelties    
            # Calculate and plot diversities
            div_warm_avg, div_warm_std = calcDiversity(
                exps=tensor_warm, k=k)
            div_cold_avg, div_cold_std = calcDiversity(
                exps=tensor_cold, k=k)
            
            nov_warm_avg_short, nov_warm_std_short = calcNovelty(
                exps=tensor_warm, k=k, longterm=False)
            nov_cold_avg_short, nov_cold_std_short = calcNovelty(
                exps=tensor_cold, k=k, longterm=False)
            
            nov_warm_avg_long, nov_warm_std_long = calcNovelty(
                exps=tensor_warm, k=k, longterm=True)
            nov_cold_avg_long, nov_cold_std_long = calcNovelty(
                exps=tensor_cold, k=k, longterm=True)
            
            plotCompare(
                avg_1=div_warm_avg, std_1=div_warm_std, 
                avg_2=div_cold_avg, std_2=div_cold_std,
                novelty=False)
            plotCompare(
                avg_1=nov_warm_avg_short, std_1=nov_warm_std_short,
                avg_2=nov_cold_avg_short, std_2=nov_cold_std_short,
                novelty=True, longterm=False)
            plotCompare(
                avg_1=nov_warm_avg_long, std_1=nov_warm_std_long,
                avg_2=nov_cold_avg_long, std_2=nov_cold_std_long,
                novelty=True, longterm=True)
                        
        if args.df: # Export dataframe
            # Collect data over all generations
            div_warm = calcDiversity(
                exps=tensor_warm, k=k, df=True)
            div_cold = calcDiversity(
                exps=tensor_cold, k=k, df=True)
            
            nov_warm_short = calcNovelty(
                exps=tensor_warm, k=k, longterm=False, df=True)
            nov_cold_short = calcNovelty(
                exps=tensor_cold, k=k, longterm=False, df=True)
            
            nov_warm_long = calcNovelty(
                exps=tensor_warm, k=k, longterm=True, df=True)
            nov_cold_long = calcNovelty(
                exps=tensor_cold, k=k, longterm=True, df=True)
            
            # Limit data to final generations and add to df
            div_warm = div_warm[:,-1]
            div_cold = div_cold[:,-1]
            
            nov_warm_short = nov_warm_short[:,-1]
            nov_cold_short = nov_cold_short[:,-1]
            
            nov_warm_long = nov_warm_long[:,-1]
            nov_cold_long = nov_cold_long[:,-1]
            
            # Insert data into DataFrames and export
            df_div = pd.DataFrame()
            df_nov = pd.DataFrame()
            
            df_div["warm"] = div_warm
            df_div["cold"] = div_cold
            
            df_nov["warm_short"] = nov_warm_short
            df_nov["cold_short"] = nov_cold_short
            df_nov["warm_long"] = nov_warm_long
            df_nov["cold_long"] = nov_cold_long
            
            dfName_div = "DIVERSITY.csv"
            dfName_nov = "NOVELTY.csv"
            if args.dfName: 
                dfName_div = args.dfName + "_" + dfName_div
                dfName_nov = args.dfName + "_" + dfName_nov
                
            df_div.to_csv(dfName_div, index=False)
            df_nov.to_csv(dfName_nov, index=False)
            print("Dataframes exported. Closing.")
            
    else: print("Pass databases with -db1 and -db2. Closing.")

if __name__ == "__main__":
    main()
