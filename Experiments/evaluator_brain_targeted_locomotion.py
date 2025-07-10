"""Evaluator class."""

import config
import math
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from pyrr import Quaternion, Vector3
from typing import Union

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkLocomotion, BrainCpgNetworkLocomotionNewstate, CpgNetworkStructure
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from revolve2.simulation.scene.geometry import GeometryPlane, GeometrySphere
from revolve2.simulation.scene import Color, Pose
from revolve2.simulation.scene.vector2 import Vector2
from revolve2.simulators.mujoco_simulator.textures import Flat

# Manually adjust font size globally
plt.rcParams.update({
    "font.size": 10,         # Default text size
    "axes.titlesize": 35,    # Title font size
    "axes.labelsize": 35,    # Axis label size
    "xtick.labelsize": 30,   # Tick label size
    "ytick.labelsize": 30,
    "legend.fontsize": 25,
    "figure.titlesize": 35,
    "figure.figsize": (15,12)
})

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
        waypointTerrain: bool = False,
        record: bool = False,
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
        :param waypointTerrain: True to render waypoints in simulator (for rerun.py).
        :param record: True to record simulation.
        """
        self._simulator = LocalSimulator(
            viewer_type = "native", headless=headless, num_simulators=num_simulators
        )
        self._cpg_network_structure = cpg_network_structure
        self._body = body
        self._output_mapping = output_mapping
        self._nose=nose
        self._targets=targets
        self._terrain = terrains.flat()
        if waypointTerrain: self._terrain = self.genTerrain()
        
    def evaluate(
        self,
        solutions: list[npt.NDArray[np.float_]], 
        sim_time: int, use_state_reset: bool,
    ) -> npt.NDArray[np.float_]:
        """
        Evaluate multiple solutions vectors for a robot.

        :param solutions: Solutions to evaluate.
        :param sim_time: Simulation time in seconds. Set to None for indefinite simulation.
        :returns: Fitnesses of the solutions.
        """

        if use_state_reset:     # Simulate with state-array resetting when chainging actions
            robots = [
                ModularRobot(
                    body=self._body,
                    brain=BrainCpgNetworkLocomotionNewstate.uniform_from_params(
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
            
        else:                   # Simulate without state-array resetting
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
        batch_parameters = make_standard_batch_parameters()
        batch_parameters.simulation_time = sim_time
        
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=batch_parameters,
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
        allCoords = []
        for robot, scenes in zip(robots, scene_states):
            coords = [
                scene.get_modular_robot_simulation_state(robot).get_pose().position for scene in scenes
                ]
            coords = np.array(coords)[:,:2] # Ignore z-coordinates
            allCoords.append(coords)
            
            fitnesses.append(self.rollBack(
                coords=coords, targets=self._targets.copy(), threshold=0.25))
            
        return fitnesses, allCoords
            
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
                    score += 1 # Score 1
                except: # No targets left to traverse to (Inshaallah)
                    score += 10 # Score 2
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
            self, coords: list[npt.NDArray[np.float_]], 
            targets: npt.NDArray[np.float_],
            plotTitle: str = "", demo: bool = False,
            ) -> None:
        """
        Plot a robot's trajectory.
        TODO: Implement drawing (un)reached targets.
        ONLY USE TO PLOT ONE SOLUTION OF A ROBOT
        :param demo: Remove colorbar for empty plot
        """
        if not demo: title=f"Targets: {targets[0]}  {targets[1]}  {targets[2]}"
        if plotTitle != "": title = plotTitle + "\n" + title
        coords = coords[0]
        x_r = coords[:,0]
        y_r = coords[:,1]
        x_t = targets[:,0]
        y_t = targets[:,1]

        plt.figure()
        # Targets
        ax = plt.gca()
        for i in range(len(targets)):
            circle = plt.Circle(targets[i], radius=0.25,
                                facecolor="pink", edgecolor="red")
            ax.add_patch(circle)    
        plt.scatter(x_t, y_t, c="r", marker="1", s=200, label="Targets")
        # Robot trajectory
        plt.plot(np.array(x_r), np.array(y_r), c="gray", alpha=0.4)
        plt.scatter(x_r, y_r, c=np.linspace(0,10,len(x_r)), s=20)
        # Start position
        plt.scatter(0, 0, c="deeppink", marker="X", s=800, label="Start pos")
        # Appearance
        plt.grid(visible=True, axis="both", ls="--")
        plt.title("Robot trajectory\n" + title)
        plt.xlabel("X")
        plt.ylabel("Y")
        if not demo: plt.colorbar(label="Time")
        plt.gca().set_aspect("equal")
        ax.set_xlim([-1,2])
        plt.legend()
        plt.show()
        
    def plotMulti(
            self, routes: npt.NDArray[np.float_], 
            figName: Union[None, str] = None
            ) -> None:
        """
        Plot multiple trajectories.
        
        :param routes: Trajectory coordinates from the best-performing robots.
        :param figName: [Optional] Figure title addition.
        """
        assert len(routes) > 1, ">1 trajectories required."
        
        fig, ax = plt.subplots(figsize=(9,11))

        # 1. Plot target background circles first (lowest layer)
        for c in config.TARGETS:
            circle = plt.Circle(c, radius=0.25,
                                facecolor="pink", edgecolor="red", lw=2, zorder=1)
            ax.add_patch(circle)
        
        # 2. Plot target points
        x_t = config.TARGETS[:, 0]
        y_t = config.TARGETS[:, 1]
        ax.scatter(x_t, y_t, c="r", s=200, marker="1", label="Targets", zorder=5)
        
        # 3. Plot all but the best route
        for route in routes[1:]:
            x = route[:, 0]
            y = route[:, 1]
            ax.plot(x, y, c="coral", lw=8, alpha=0.4, zorder=30)
        
        # 4. Plot best route
        x = routes[0][:, 0]
        y = routes[0][:, 1]
        ax.plot(x, y, c="gray", lw=7, alpha=0.5, zorder=35)
        plt.scatter(x, y, c=np.linspace(0, config.SIM_TIME, len(x)), s=60, zorder=40)
        
        # 5. Start position
        ax.scatter(0, 0, c="deeppink", marker="X", s=800, label="Start pos", zorder=50)
        
        # Final adjustments
        title = "Robot trajectories"
        if figName != None: title += ("\n" + figName)
        plt.title(title)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim([-1,2])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(visible=True, axis="both", ls="--")
        plt.colorbar(label="Time [s]")
        ax.legend()
        plt.show()

    def genTerrain(self) -> Terrain:
        """
        Generate terrain with targets for rerun visualization.
        :param targets: List of xy-coordinates of targets/waypoints.
        """
        # Plane
        static_geometry = [
            GeometryPlane(
                pose=Pose(),
                mass=0.0,
                size=Vector2([20.0,20.0]),
                )
            ]
        
        # Waypoint spheres
        targets = config.TARGETS
        targets = targets*-1
        targetSpheres = [
            GeometrySphere(
                pose=Pose(
                    position=Vector3([t[0], t[1], -0.98]), orientation=Quaternion()
                    ),
                mass=0.0,
                radius=1.0,
                texture=Flat(primary_color=Color(255, 0, 200, 255)),
                ) for t in targets
            ]
        
        static_geometry += targetSpheres
        
        return Terrain(static_geometry=static_geometry)
    