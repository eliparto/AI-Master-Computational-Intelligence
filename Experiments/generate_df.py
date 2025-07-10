""" Generate dataframes for statistical analysis (mean and max fitnesses). """

import numpy as np
import pandas as pd
from pandas import DataFrame
import argparse
import json
from database_components import Experiment, Generation, Individual, Population
from sqlalchemy import select
from revolve2.experimentation.database import OpenMethod, open_database_sqlite

def genDF(fileName) -> DataFrame:
    """
    Generates a dataframe for an experiment.
    """
    dbengine = open_database_sqlite(
        fileName, open_method=OpenMethod.OPEN_IF_EXISTS
    )

    df = pd.read_sql(
        select(
            Experiment.id.label("experiment_id"),
            Generation.generation_index,
            Individual.fitness,
        )
        .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
        .join_from(Generation, Population, Generation.population_id == Population.id)
        .join_from(Population, Individual, Population.id == Individual.population_id),
        dbengine,
    )
    
    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Output relevant .csv files for statistical analysis.")
    parser.add_argument("-name", type=str, help="Specify the name of the JSON containing databases to combine.")
    parser.add_argument("-output", type=str, help="Specify the name of the output file.", default="temp")
    args = parser.parse_args()
    
    # Import JSON file containing database names
    dbList = args.name + ".json"
    with open(dbList, 'r') as file:
        dbFile = json.load(file)    
    dbs = dbFile.get("Databases", [])
    assert len(dbs) > 0, "Need to import at least 2 databases to combine."
    
    # Read first database in list
    data = genDF("Databases/"+dbs[0]+".sqlite")
    dbs = dbs[1:]
    
    # Read next databases and combine into one
    for fileName in dbs:
        df = genDF("Databases/"+fileName+".sqlite")
        df.experiment_id += data.experiment_id.max()
        data = pd.concat([data, df], ignore_index=True)

    # Generate and export output dataframe
    max_exp = data.experiment_id.max()
    max_gen = data.generation_index.max()    
    print(f"Total of {data.experiment_id.max()} experiments aggregated.")
        
    fit_mean = [
        data[
            (data["experiment_id"] == i)&(data["generation_index"] == max_gen)
        ].fitness.mean() for i in range(1, max_exp+1)
    ]
    fit_max = [
        data[
            (data["experiment_id"] == i)&(data["generation_index"] == max_gen)
        ].fitness.max() for i in range(1, max_exp+1)
    ]
    dfOut = pd.DataFrame()
    dfOut["experiment"] = np.arange(1, max_exp+1, 1)
    dfOut["fit_mean"] = fit_mean
    dfOut["fit_max"] = fit_max
    
    fileName = args.output + ".csv"
    dfOut.to_csv(fileName, index=False)
    
if __name__ == "__main__":
    main()
