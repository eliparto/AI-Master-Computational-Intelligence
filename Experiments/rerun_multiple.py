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
            fitnesses = [row[3] for row in row_list]
            solutions = [row[4] for row in row_list]
            noses = [row[5] for row in row_list]
            rows_by_experiment_and_generation[experiment_id][generation_index] = (
                genotypes, fitnesses, solutions, noses
                )

    experiments = [
            [
            rows_by_experiment_and_generation[i][j] for j in range(config.NUM_GENERATIONS_BODY+1)
            ] for i in range(1, len(rows_by_experiment_and_generation)+1)
        ]

    # Extract bets performing genotypes and solutions per experiment
    robots = [
        [
            e[-1][i][-1] for i in range(4)
            ] for e in experiments
        ]
    
    return robots

def main() -> None:
    """Perform the rerun."""
    # Check for passed arguments
    parser = argparse.ArgumentParser(description="Plot the n best trajectories for a test condition.")
    parser.add_argument("-name", type=str, help="Specify the database JSON filename.")
    parser.add_argument("-figName", type=str, default = None, help="Specify figure title.")
    parser.add_argument("-t", "-time", type=int, default=120, help="Specify simulation time. Set to '0' for indefinite time. Default: 120.")
    parser.add_argument("-n", type=int, default=3, help="Specify no. of best trajectories n to plot.")
    args = parser.parse_args()
        
    if args.name:
        assert args.t > 0, "Invalid simulation time passed."
        
        # Import JSON file containing database names
        dbList = args.name + ".json"
        with open(dbList, 'r') as file:
            dbFile = json.load(file)    
        dbs = dbFile.get("Databases", [])
        assert len(dbs) > 0, "Need to import at least 2 databases to combine."
        
        # Extract best robot's genotypes, fitnesses, and solutions
        robots = []
        for db in dbs:
            dbName = "Databases/" + db + ".sqlite"
            robots += extract(dbName)
            
        # Rerun to collect trjectories
        print("Simulating best robots...")
        simFitnesses = []
        fitnesses = []
        trajectories = []
        for (genotype, fitness, solutions, nose) in tqdm(robots):
            # Generate body and brain
            body = genotype.develop().body
            active_hinges = body.find_modules_of_type(ActiveHinge)
            (
                cpg_network_structure, 
                output_mapping
            ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
            
            targets = config.TARGETS
            evaluator = Evaluator(
                headless=True,
                num_simulators=1,
                cpg_network_structure=cpg_network_structure,
                output_mapping=output_mapping,
                body=body,
                targets=targets,
                nose=nose,
                waypointTerrain=False,
                )
            
            simFitness, coords = evaluator.evaluate(
                solutions = [solutions],
                sim_time = 120,
                use_state_reset=True
                )
            
            simFitnesses.append(simFitness[0]) # Sanity check
            fitnesses.append(fitness)
            trajectories.append(coords)
        
        # Convert to np arrays and sort by training fitness
        # Rerun fitness might differ due to simulator synchronization mismatches
        simFitnesses = np.array(simFitnesses)
        fitnesses = np.array(fitnesses)
        trajectories = np.array(trajectories)
        # Reshape trajectory array
        tr = trajectories.shape
        trajectories = np.reshape(trajectories, (tr[0], tr[2], tr[3]))
        # Sort by fitness (descending)
        idx = np.flip(np.argsort(simFitnesses))
        trajectories = trajectories[idx]
        
        print("Simulation results:")
        print(f"Training:\n{simFitnesses[idx]}\n")
        print(f"Rerun:\n{fitnesses[idx]}\n")
        print("Sorted by fitness (rerun):")
        print(fitnesses[idx])
        # Plot n best individuals
        evaluator.plotMulti(routes=trajectories[:args.n], figName=args.figName)
        
    else: print("Pass database name with '-name'. Closing now.")

if __name__ == "__main__":
    main()
