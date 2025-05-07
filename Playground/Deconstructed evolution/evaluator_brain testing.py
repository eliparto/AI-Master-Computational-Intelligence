"""Evaluator class."""

import math

import numpy as np
import numpy.typing as npt

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic, BrainCpgNetworkLocomotion, CpgNetworkStructure
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

from revolve2.standards.fitness_functions import FitnessEvaluator

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
    ) -> None:
        """
        Initialize this object.

        :param headless: `headless` parameter for the physics simulator.
        :param num_simulators: `num_simulators` parameter for the physics simulator.
        :param cpg_network_structure: Cpg structure for the brain.
        :param body: Modular body of the robot.
        :param output_mapping: A mapping between active hinges and the index of their corresponding cpg in the cpg network structure.
        """
        self._simulator = LocalSimulator(
            viewer_type = "native", headless=headless, num_simulators=num_simulators
        )
        self._terrain = terrains.flat()
        self._cpg_network_structure = cpg_network_structure
        self._body = body
        self._output_mapping = output_mapping

    def evaluate(
        self,
        solutions: list[npt.NDArray[np.float_]], fit_type: int
    ) -> npt.NDArray[np.float_]:
        """
        Evaluate multiple robots.

        :param solutions: Solutions to evaluate.
        :fit_type: Integer determining fitness type to calculate.
        :returns: Fitnesses of the solutions.
        """
        
        # TODO: Change to the locomotion network
        # TODO: Use a single robot instead of multiple
        # TODO: Figure out where to put target location and controller

        robots = [
            ModularRobot(
                body=self._body,
                brain=BrainCpgNetworkStatic.uniform_from_params(
                    params=params,
                    cpg_network_structure=self._cpg_network_structure,
                    initial_state_uniform=math.sqrt(2) * 0.5,
                    output_mapping=self._output_mapping,
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

        # xy_displacements = [
        #     fitness_functions.xy_displacement(
        #         states[0].get_modular_robot_simulation_state(robot),
        #         states[-1].get_modular_robot_simulation_state(robot),
        #     )
        #     for robot, states in zip(robots, scene_states)
        # ]

        fits_forward, betas = self.calcFitForward(robots, scene_states)
        fits_rot_l = self.calcFitRotation(robots, scene_states)
        
        if fit_type == 0: fits = fits_forward
        elif fit_type == 1: fits = fits_rot_l - fits_forward
        else: fits = -fits_rot_l - fits_forward
        
        return fits, betas
        # TODO: Compare xy-displacement to generated fitness array
    
    def calcFitForward(
        self, robots: ModularRobot, scene_states) -> list[list[float], float]:
        """
        Calculate robot displacements and natural orientation angles beta.
        :robots: Robots to be evaluated.
        :scene_states: Simulated scened for every robot.
        """
        fitnesses = []
        betas = []
        for robot, states in zip(robots, scene_states):
            sim_state_begin = states[0].get_modular_robot_simulation_state(robot)
            sim_state_end = states[-1].get_modular_robot_simulation_state(robot)
            
            # Start and end position vectors
            begin_pos = np.array([
                sim_state_begin.get_pose().position.x,
                sim_state_begin.get_pose().position.y])
            end_pos = np.array([
                sim_state_end.get_pose().position.x, 
                sim_state_end.get_pose().position.y
                ])
            
            print("\nDisplacement calculation:")
            print(end_pos)
         
            # Positional displacement vector
            disp = end_pos - begin_pos
            fitness = np.linalg.norm(disp)
            
            # Robot's natural orientation
            beta = np.arctan2(disp[0], disp[1])
            
            fitnesses.append(fitness)
            betas.append(beta)
            
        return np.array(fitnesses), np.array(betas)
    
    def calcFitRotation(
        self, robots: ModularRobot, scene_states) -> npt.NDArray[np.float_]:
        """
        Calculate the yaw angles (rotation around z-axis) of all robots.
        :robots: Robots to be evaluated.
        :scene_states: Simulated scened for every robot.
        """
        fitnesses = []
        for robot, states in zip(robots, scene_states):
            fitness = 0.0
            for idx in range(len(states)-1):
                # Extract simulation states at time t and (t+1)
                sim_state_t = states[idx].get_modular_robot_simulation_state(robot)
                sim_state_t_1 = states[idx+1].get_modular_robot_simulation_state(robot)
                # Extract quaternions
                quat_t = sim_state_t.get_pose().orientation
                quat_t_1 = sim_state_t_1.get_pose().orientation
                # Convert quaternion data to yaw angles
                _, _, yaw_start = self.quaternion_to_euler(quat_t)
                _, _, yaw_end = self.quaternion_to_euler(quat_t_1)
                
                # Calculate delta theta and pass through low-pass filter
                delta = yaw_end - yaw_start
                if abs(delta) > np.pi: delta = 0
                fitness += delta
            fitnesses.append(fitness)
            
        return np.array(fitnesses)
                
    def quaternion_to_euler(self, q) -> tuple[float]:
        """
        Convert quaterion data into angles about roll, pitch and yaw axes.
        """
        w, x, y, z = q
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    
        return roll, pitch, yaw  # Angles in radians
        
        













