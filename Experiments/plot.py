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

def plotFitness(df, name, figName, fMax, fMin):
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
        lw=4,
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
        lw=4,
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

    title = f"Mean and max {name} across repetitions with std as shade"
    if figName != "": title += ("\n" + figName)
    plt.xlabel("Generation index")
    plt.ylabel("Fitness (no. of targets reached)")
    ax = plt.gca()
    #ax.set_xlim(0, config.NUM_GENERATIONS_BODY)
    ax.set_xlim(0,10) # Temporary
    ax.set_ylim(fMin, fMax)
    plt.grid(which="major", axis="both")
    plt.title(title)
    plt.legend()

def plotLearningDelta(df, figName, fMax, fMin):
    """Plot learning delta (fitness - old_fitness) over generations."""
    # Calculate learning delta
    df['learning_delta'] = df['fitness'] - df['old_fitness']
    
    agg_per_experiment_per_generation = (
        df.groupby(["experiment_id", "generation_index"])
        .agg({"learning_delta": ["max", "mean"]})
        .reset_index()
    )
    agg_per_experiment_per_generation.columns = [
        "experiment_id",
        "generation_index",
        "max_delta",
        "mean_delta",
    ]

    agg_per_generation = (
        agg_per_experiment_per_generation.groupby("generation_index")
        .agg({"max_delta": ["mean", "std"], "mean_delta": ["mean", "std"]})
        .reset_index()
    )
    agg_per_generation.columns = [
        "generation_index",
        "max_delta_mean",
        "max_delta_std",
        "mean_delta_mean",
        "mean_delta_std",
    ]
    
    plt.figure()

    # Plot max learning delta
    plt.plot(
        agg_per_generation["generation_index"],
        agg_per_generation["max_delta_mean"],
        label="Max learning delta",
        color="g",
        lw=4,
    )
    plt.fill_between(
        agg_per_generation["generation_index"],
        agg_per_generation["max_delta_mean"] - agg_per_generation["max_delta_std"],
        agg_per_generation["max_delta_mean"] + agg_per_generation["max_delta_std"],
        color="g",
        alpha=0.2,
    )

    # Plot mean learning delta
    plt.plot(
        agg_per_generation["generation_index"],
        agg_per_generation["mean_delta_mean"],
        label="Mean learning delta",
        color="orange",
        lw=4,
    )
    plt.fill_between(
        agg_per_generation["generation_index"],
        agg_per_generation["mean_delta_mean"]
        - agg_per_generation["mean_delta_std"],
        agg_per_generation["mean_delta_mean"]
        + agg_per_generation["mean_delta_std"],
        color="orange",
        alpha=0.2,
    )

    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)

    title = "Mean and max learning delta (fitness - old_fitness) across repetitions with std as shade"
    if figName != "": title += ("\n" + figName)
    plt.xlabel("Generation index")
    plt.ylabel("Learning Delta (fitness improvement)")
    ax = plt.gca()
    #ax.set_xlim(0, config.NUM_GENERATIONS_BODY)
    ax.set_xlim(0,10) # Temporary
    ax.set_ylim(fMin, fMax)
    plt.grid(which="major", axis="both")
    plt.title(title)
    plt.legend()

def main() -> None:
    """Run the program."""
    # Check for passed arguments
    parser = argparse.ArgumentParser(description="Plot the max and mean fitnesses for an experiment.")
    parser.add_argument("-name", type=str, help="Specify the input database's filename.")
    parser.add_argument("-figName", type=str, help="Specify custom figure title.")
    parser.add_argument("-fMax", type=float, help="Specify the max value of the y (fitness) axis.", default=3.0)
    parser.add_argument("-fMin", type=float, help="Specify the min value of the y (fitness) axis.", default=-10.0)
    parser.add_argument("-plotDelta", action="store_true", help="Plot learning delta (fitness - old_fitness) instead of absolute fitness.")
    args = parser.parse_args()
    
    if args.name:
        dbName = "Databases/" + args.name + ".sqlite"
        dbengine = open_database_sqlite(
            dbName, open_method=OpenMethod.OPEN_IF_EXISTS
        )
    
        # Modify SQL query to include old_fitness if plotting delta
        if args.plotDelta:
            df = pd.read_sql(
                select(
                    Experiment.id.label("experiment_id"),
                    Generation.generation_index,
                    Individual.fitness,
                    Individual.old_fitness,
                )
                .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
                .join_from(Generation, Population, Generation.population_id == Population.id)
                .join_from(Population, Individual, Population.id == Individual.population_id),
                dbengine,
            )
        else:
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

        if args.figName: 
            figName = args.figName
        else: 
            figName = ""
        
        # Choose which plot to generate
        if args.plotDelta:
            # Check if old_fitness column exists and has non-null values
            if 'old_fitness' not in df.columns:
                print("Error: old_fitness column not found in database. Cannot plot learning delta.")
                return
            if df['old_fitness'].isnull().all():
                print("Warning: old_fitness column contains only null values. Cannot plot learning delta.")
                return
            plotLearningDelta(df, figName, args.fMax, args.fMin)
        else:
            plotFitness(df, "fitness", figName, args.fMax, args.fMin)
        
        plt.show()
        
    else: 
        print("Pass database name with '-name'. Closing now.")

if __name__ == "__main__":
    main()