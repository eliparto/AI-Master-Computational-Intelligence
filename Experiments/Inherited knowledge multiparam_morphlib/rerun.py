"""Rerun the best robot between all experiments."""

import logging
import numpy as np
from matplotlib import pyplot as plt
import time

from database_components import Genotype, Individual
from evaluator_brain import Evaluator
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

def main() -> None:
    """Perform the rerun."""
    # Check for passed arguments
    parser = argparse.ArgumentParser(description="Show the best performing robot for a given evolutionary database.")
    parser.add_argument("-name", type=str, help="Specify the database filename.")
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
                select(Genotype, Individual.fitness, Individual.fitnesses, 
                       Individual.solutions)
                .join_from(Genotype, Individual, Genotype.id == Individual.genotype_id)
                .order_by(Individual.fitness.desc())
                .limit(1)
            ).one()
            assert row is not None
    
            # Retrieve necessary information and reshape weight vectors
            genotype = row[0]
            fitness = row[1]
            fitnesses = row[2]
            solutions = row[3]
            solutions = np.reshape(solutions, (3, int(len(solutions)/3)))
    
        logging.info(f"Best fitness: {fitness}")
        
        # Generate the robot's body and brain
        body = genotype.develop().body
        active_hinges = body.find_modules_of_type(ActiveHinge)
        brain = (
            cpg_network_structure,
            output_mapping,
        ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
        
        # Setup the evaluator
        evaluator = Evaluator(
        headless=False,
        num_simulators=1,
        cpg_network_structure=cpg_network_structure,
        body=body,
        output_mapping=output_mapping,
        )
        
        # Simulate robot with the 3 solution vectors
        motions = ["Forward", "Rotate left", "Rotate right"]
        for i in range(3):
            print(f"Motion: {motions[i]}")
            print(f"{motions[i]} fitness: {round(fitnesses[i], 4)}")
            evaluator.evaluate([solutions[i]], 0)
            time.sleep(2)
        
        
    else: print("Pass database name with '-name'. Closing now.")
    

if __name__ == "__main__":
    main()
