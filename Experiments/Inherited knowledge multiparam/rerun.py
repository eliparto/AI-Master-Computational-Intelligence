"""Rerun the best robot between all experiments."""

import logging
import numpy as np
from matplotlib import pyplot as plt
import time
import config

from database_components import Genotype, Individual
from evaluator_brain_targeted_locomotion import Evaluator
from sqlalchemy import select
from sqlalchemy.orm import Session
import argparse

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)

from BrainOptimizer import BrainOptimizerDE as opt

def main() -> None:
    """Perform the rerun."""
    # Check for passed arguments
    parser = argparse.ArgumentParser(description="Show the best performing robot for a given evolutionary database.")
    parser.add_argument("-name", type=str, help="Specify the database filename.")
    parser.add_argument("-t", "-time", type=int, help="Specify simulation time. Set to '0' for indefinite time.")
    parser.add_argument("-p", "-plot", action="store_true", help="Toggle to disable simulator view and only plot trajectory.")
    args = parser.parse_args()
    
    setup_logging()
    
    if args.name:
        # Database name
        db_name = "Databases/" + args.name + ".sqlite"
        
        # Load the best individual from the database.
        dbengine = open_database_sqlite(
            db_name, open_method=OpenMethod.OPEN_IF_EXISTS
        )
    
        with Session(dbengine) as ses:
            row = ses.execute(
                select(Genotype, Individual.fitness, Individual.solutions, 
                       Individual.nose)
                .join_from(Genotype, Individual, Genotype.id == Individual.genotype_id)
                .order_by(Individual.fitness.desc())
                .limit(1)
            ).one()
            assert row is not None
    
            # Retrieve necessary information and reshape weight vectors
            genotype = row[0]
            fitness = row[1]
            solutions = row[2]
            nose = row[3]

        # Generate the robot's body and brain
        body = genotype.develop().body
        active_hinges = body.find_modules_of_type(ActiveHinge)
        (
            cpg_network_structure,
            output_mapping,
        ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
        
        # Setup the evaluator
        bounds = (-5,5)
        targets = opt(bounds).generateTargets()

        if args.p: headless = True # Disable mujoco viewer to plot trajectory
        else: headless = False
        
        print("Simulating...")
        evaluator = Evaluator(
            headless=headless,
            num_simulators=1,
            cpg_network_structure=cpg_network_structure,
            output_mapping=output_mapping,
            body=body,
            targets=targets,
            nose=nose,
            )
        
        sim_time = args.t
        if sim_time == 0: sim_time = None
        simFitness, coords = evaluator.evaluate(
            solutions=[solutions],
            sim_time=sim_time,
            )
        print("Simulating done.")
        print(f"Targets:\n{targets[:3]}")
        print(f"Training fitness:\t{fitness}")
        print(f"Rerun fitness:\t\t{simFitness[0]}")
        if args.p: evaluator.plotTrajectory(coords, config.TARGETS)
        
    else: print("Pass database name with '-name'. Closing now.")

if __name__ == "__main__":
    main()
