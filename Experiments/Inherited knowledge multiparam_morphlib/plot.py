"""Plot fitness over generations for all experiments, averaged
Decompose fitness vectors w/:
df[["fitness_forward", "fitness_rot_left", "fitness_rot_right"]] = pd.Dataframe(df["fitnesses"].tolist(), index = df.index)
"""

import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from database_components import Experiment, Generation, Individual, Population
from sqlalchemy import select
import argparse

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging

def plotFitness(df, name):
    agg_per_experiment_per_generation = (
        df.groupby(["experiment_id", "generation_index"])
        .agg({name: ["max", "mean"]})
        .reset_index()
    )
    agg_per_experiment_per_generation.columns = [
        "experiment_id",
        "generation_index",
        "max_fitness",
        "mean_fitness",
    ]

    agg_per_generation = (
        agg_per_experiment_per_generation.groupby("generation_index")
        .agg({"max_fitness": ["mean", "std"], "mean_fitness": ["mean", "std"]})
        .reset_index()
    )
    agg_per_generation.columns = [
        "generation_index",
        "max_fitness_mean",
        "max_fitness_std",
        "mean_fitness_mean",
        "mean_fitness_std",
    ]
    
    plt.figure()

    # Plot max
    plt.plot(
        agg_per_generation["generation_index"],
        agg_per_generation["max_fitness_mean"],
        label="Max fitness",
        color="b",
    )
    plt.fill_between(
        agg_per_generation["generation_index"],
        agg_per_generation["max_fitness_mean"] - agg_per_generation["max_fitness_std"],
        agg_per_generation["max_fitness_mean"] + agg_per_generation["max_fitness_std"],
        color="b",
        alpha=0.2,
    )

    # Plot mean
    plt.plot(
        agg_per_generation["generation_index"],
        agg_per_generation["mean_fitness_mean"],
        label="Mean fitness",
        color="r",
    )
    plt.fill_between(
        agg_per_generation["generation_index"],
        agg_per_generation["mean_fitness_mean"]
        - agg_per_generation["mean_fitness_std"],
        agg_per_generation["mean_fitness_mean"]
        + agg_per_generation["mean_fitness_std"],
        color="r",
        alpha=0.2,
    )

    plt.xlabel("Generation index")
    plt.ylabel("Fitness")
    plt.title("Mean and max " + name + " across repetitions with std as shade")
    plt.legend()

def plotSolution(solution, name):
    plt.figure()
    plt.hist(solution)
    plt.xlabel("Weight value")
    plt.ylabel("Frequency")
    plt.title(name)
    

def main() -> None:
    """Run the program."""
    # Check for passed arguments
    parser = argparse.ArgumentParser(description="Plot the max and mean fitnesses for an experiment.")
    parser.add_argument("-name", type=str, help="Specify the input database's filename.")
    args = parser.parse_args()
    
    if args.name:
        dbName = "Databases/" + args.name + ".sqlite"
        dbengine = open_database_sqlite(
            dbName, open_method=OpenMethod.OPEN_IF_EXISTS
        )
    
        df = pd.read_sql(
            select(
                Experiment.id.label("experiment_id"),
                Generation.generation_index,
                Individual.fitness,
                Individual.fitnesses,
                Individual.solutions,
            )
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id),
            dbengine,
        )
        df[["fitness_forward", "fitness_rot_left", "fitness_rot_right"]] = pd.DataFrame(
            df["fitnesses"].tolist(), index = df.index)
        df.drop(columns = ["fitnesses"], inplace = True)
        
        solutions = df.loc[df["fitness"].idxmax(), "solutions"]
        solutions = np.reshape(solutions, (3, int(len(solutions)/3)))
        
        #for col_name in fit_types:
        plotFitness(df, "fitness")
        plotFitness(df, "fitness_forward")
        plotFitness(df, "fitness_rot_left")
        plotFitness(df, "fitness_rot_right")
        plotSolution(solutions[0], "Weights forward")
        plotSolution(solutions[1], "Weights rot left")
        plotSolution(solutions[2], "Weights rot right")
        plt.show()
        
    else: print("Pass database name with '-name'. Closing now.")

if __name__ == "__main__":
    main()
