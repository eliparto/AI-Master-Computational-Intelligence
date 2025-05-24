"""Evaluator class."""

import math

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkLocomotion, CpgNetworkStructure
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

class Evaluator:
    """Provides evaluation of robots."""

    _simulator: LocalSimulator
    _terrain: Terrain
    _cpg_network_structure: CpgNetworkStructure
    _body: Body
    _output_mapping: list[tuple[int, ActiveHinge]]

    def __init__(
        self,
        headless: bool,
        num_simulators: int,
        cpg_network_structure: CpgNetworkStructure,
        body: Body,
        output_mapping: list[tuple[int, ActiveHinge]],
        targets: npt.NDArray[np.float_],
        nose: int,
    ) -> None:
        """
        Initialize this object.

        :param headless: `headless` parameter for the physics simulator.
        :param num_simulators: `num_simulators` parameter for the physics simulator.
        :param cpg_network_structure: Cpg structure for the brain.
        :param body: Modular body of the robot.
        :param output_mapping: A mapping between active hinges and the index of their corresponding cpg in the cpg network structure.
        :param targets: List of xy-coordinates of targets for the robot to navigate towards.
        :param nose: Frontal (nose) orientation of the robot.
        """
        self._simulator = LocalSimulator(
            viewer_type = "native", headless=headless, num_simulators=num_simulators
        )
        self._terrain = terrains.flat()
        self._cpg_network_structure = cpg_network_structure
        self._body = body
        self._output_mapping = output_mapping
        self._nose=nose
        self._targets=targets

    def evaluate(
        self,
        solutions: list[npt.NDArray[np.float_]],
    ) -> npt.NDArray[np.float_]:
        """
        Evaluate multiple solutions vectors for a robot.

        :param solutions: Solutions to evaluate.
        :returns: Fitnesses of the solutions.
        """

        robots = [
            ModularRobot(
                body=self._body,
                brain=BrainCpgNetworkLocomotion.uniform_from_params(
                    params=params,
                    cpg_network_structure=self._cpg_network_structure,
                    initial_state_uniform=math.sqrt(2) * 0.5,
                    output_mapping=self._output_mapping,
                    nose=self._nose,
                    targets=self._targets,
                ),
            )
            for params in solutions
        ]

        # Create the scenes.
        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain=self._terrain)
            scene.add_robot(robot)
            scenes.append(scene)

        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scenes,
        )

        return self.calculateFitness(robots, scene_states)
        
    def calculateFitness(
            self, robots: ModularRobot, scene_states
            ) -> npt.NDArray[np.float_]:
        """
        Wrapper to calculate each robot's fitness by rolling back through their respective trajectories
        and observing how many targets have been reached.
        """
        fitnesses = []
        for robot, scenes in zip(robots, scene_states):
            coords = [
                scene.get_modular_robot_simulation_state(robot).get_pose().position for scene in scenes
                ]
            coords = np.array(coords)[:,:2] # Ignore z-coordinates
            
            fitnesses.append(self.rollBack(
                coords=coords, targets=self._targets.copy(), threshold=0.5*2**0.5))
            
        return fitnesses
            
    def rollBack(
            self, coords: npt.NDArray[np.float_], 
            targets: npt.NDArray[np.float_], threshold: float
            ) -> float:
        """
        Calculate an individual robot's fitness by retracing its steps and observing if targets have been reached.
        Points are awarded for reaching/finishing reaching all targets, and the
        last distance between the robot and its next target.
        
        :param coords: xy-coordinates of a robot.
        :targets: Target coordinates.
        """
        score = 0.0
        last_target = np.zeros(2) # In case no target is reached (first generations)
        
        # Calculate scores 1 and 2 for reaching targets
        for coord in coords:
            vect_toTarget = targets[0] - coord
            if np.linalg.norm(vect_toTarget) < threshold: # Robot is within range of target
                try: # Targets left in target list
                    last_target = targets[0]
                    targets = targets[1:] # Pop reached target
                    score += 10 # Score 1
                except: # No targets left to traverse to (Inshaallah)
                    score += 100 # Score 2
                    break
                
        # Calculate score 3 (distance to last target)
        if len(targets) >= 1: # Only possible if there are still targets left
            vect_toTarget = targets[0] - coords[-1]
            vect_targetToTarget = targets[0] - last_target
            
            # Score 3 equal to proportion of distance traveled to next target
            dist_toTarget = np.linalg.norm(vect_toTarget)
            dist_interTarget = np.linalg.norm(vect_targetToTarget)
            if dist_toTarget == 0: dist_toTarget += 0.001 # Prevent division by zero
            if dist_interTarget == 0: dist_interTarget += 0.001
            
            score += 1 - (dist_toTarget / dist_interTarget)
                
        return score
        
    def plotTrajectory(
            self, coords: npt.NDArray[np.float_], 
            targets: npt.NDArray[np.float_]
            ) -> None:
        """
        Plot a robot's trajectory.
        TODO: Implement drawing (un)reached targets.
        """
        x_r = coords[:,0]
        y_r = coords[:,1]
        x_t = targets[:,0]
        y_t = targets[:,1]

        plt.figure()
        # Robot trajectory
        plt.scatter(x_r, y_r, c=np.linspace(0,10,len(x_r)), s=20)
        # Targets
        plt.scatter(x_t, y_t, c="r", marker="1", s=50)
        # Appearance
        plt.title("Robot trajectory") # TODO: Implement robot's index
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar()
        plt.show()
    