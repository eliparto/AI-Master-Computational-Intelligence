"""Rerun the best robot between all experiments."""

import logging

from database_components import Genotype, Individual
from evaluator import Evaluator
from sqlalchemy import select
from sqlalchemy.orm import Session
import argparse

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging


def main() -> None:
    """Perform the rerun."""
    setup_logging()
    
    # Check for passed arguments
    parser = argparse.ArgumentParser(description="Show the best performing robot for a given evolutionary database.")
    parser.add_argument("-name", type=str, help="Specify the database filename.")
    args = parser.parse_args()
    
    if args.name:
        # Database name
        db_name = "Databases/" + args.name
        
        # Load the best individual from the database.
        dbengine = open_database_sqlite(
            db_name, open_method=OpenMethod.OPEN_IF_EXISTS
        )
    
        with Session(dbengine) as ses:
            row = ses.execute(
                select(Genotype, Individual.fitness)
                .join_from(Genotype, Individual, Genotype.id == Individual.genotype_id)
                .order_by(Individual.fitness.desc())
                .limit(1)
            ).one()
            assert row is not None
    
            genotype = row[0]
            fitness = row[1]
    
        logging.info(f"Best fitness: {fitness}")
    
        # Create the evaluator.
        evaluator = Evaluator(headless=False, num_simulators=1)
    
        # Show the robot.
        evaluator.evaluate([genotype])
        
    else: print("Pass database name with '-name'. Closing now.")
    

if __name__ == "__main__":
    main()
