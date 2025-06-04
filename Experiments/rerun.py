"""Rerun the best robot between all experiments."""

import config
import numpy as np
import numpy.typing as npt

from database_components import Genotype, Individual
from evaluator_brain_targeted_locomotion import Evaluator
from sqlalchemy import select
from sqlalchemy.orm import Session
import argparse

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulation.scene import AABB, Color, Pose
from revolve2.simulation.scene.geometry import GeometryHeightmap, GeometryPlane, GeometryBox, GeometrySphere
from revolve2.simulation.scene.vector2 import Vector2
from pyrr import Quaternion, Vector3
from revolve2.simulators.mujoco_simulator.textures import Checker, Flat, Gradient

def main() -> None:
    """Perform the rerun."""
    # Check for passed arguments
    parser = argparse.ArgumentParser(description="Show the best performing robot for a given evolutionary database.\nRun using mjpython on Mac for sim visualization.")
    parser.add_argument("-name", type=str, help="Specify the database filename.")
    parser.add_argument("-figName", type=str, help="Specify figure title.")
    parser.add_argument("-t", "-time", type=int, default=80, help="Specify simulation time. Set to '0' for indefinite time. Default: 80.")
    parser.add_argument("-p", action="store_true", help="Toggle to disable simulator view and only plot trajectory.")
    parser.add_argument("-e", action="store_true", help="Add to simulate using state-resetting.")
    args = parser.parse_args()
        
    if args.name:
        # Quick logic
        if args.figName: # Make sure that the script outputs a plot
            figTitle = args.figName
            assert args.p == True, "figName can only be used when plotting using -p."
        else: figTitle = ""
        if args.p: 
            headless = True # Disable viewer if cript outputs a plot
            assert args.t != 0, "Can't plot path with indefinite simulation time"
        else: headless = False
        
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
        # bounds = (-5,5)
        # targets = opt(bounds).generateTargets()
        targets = config.TARGETS
        
        print("Simulating...")

        evaluator = Evaluator(
            headless=headless,
            num_simulators=1,
            cpg_network_structure=cpg_network_structure,
            output_mapping=output_mapping,
            body=body,
            targets=targets,
            nose=nose,
            waypointTerrain=True,
            )
        
        sim_time = args.t
        if sim_time == 0: sim_time = None # Indefinite simulation time
        
        simFitness, coords = evaluator.evaluate(
            solutions=[solutions],
            sim_time=sim_time,
            use_state_reset=args.e
            )
        print("Simulating done.")
        print(f"Training fitness:\t{fitness}")
        print(f"Rerun fitness:\t\t{simFitness[0]}")
        if args.p: evaluator.plotTrajectory(coords, config.TARGETS, figTitle)
        
    else: print("Pass database name with '-name'. Closing now.")

if __name__ == "__main__":
    main()
