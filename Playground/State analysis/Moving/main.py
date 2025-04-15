"""Main script for the example."""

import logging
import numpy as np
from matplotlib import pyplot as plt

from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighborRandom
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import fitness_functions, modular_robots_v2, terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

# Modified fitness calculator
from revolve2.standards.fitness_functions import FitnessEvaluator


"""Run the simulation."""
# Setup
setup_logging()
rng = make_rng_time_seed()

# Create the robot.
body = modular_robots_v2.gecko_v2()
brain = BrainCpgNetworkNeighborRandom(body=body, rng=rng)
robot = ModularRobot(body, brain)

# Create the scene.
scene = ModularRobotScene(terrain=terrains.flat())
scene.add_robot(robot)

# Create the simulator.
# We set enable the headless flag, which will prevent visualization of the simulation, speeding it up.
simulator = LocalSimulator(viewer_type="native", headless=True)
batch_parameters = make_standard_batch_parameters()
batch_parameters.simulation_time = 60

# Obtain the state of the simulation, measured at a predefined interval as defined in the batch parameters.
scene_states = simulate_scenes(
    simulator=simulator,
    batch_parameters=batch_parameters,
    scenes=scene,
)

"""
Using the previously obtained scene_states we can now start to evaluate our robot.
Note in this example we simply use x-y displacement, but you can do any other way of evaluation as long as the required data is in the scene states.
"""
# Get the state at the beginning and end of the simulation.
# scene_state_begin = scene_states[0]
# scene_state_end = scene_states[-1]

# Retrieve the state of the modular robot, which also contains the location of the robot.
# robot_state_begin = scene_state_begin.get_modular_robot_simulation_state(robot)
# robot_state_end = scene_state_end.get_modular_robot_simulation_state(robot)

# Convert scenes into states
sim_states = [
    scene.get_modular_robot_simulation_state(robot) for scene in scene_states
    ]

# Set up fitness evaluator and calculate fitnesses
Fit_eval = FitnessEvaluator(sim_states)
fit_v, beta = Fit_eval.xy_displacement()
fit_rot, deltas_pure, deltas_filtered, orients = Fit_eval.rotation()
positions = Fit_eval.displacement()

print(f"Displacement: {fit_v}")
print(f"Rotation: {fit_rot}")

# Plot movements
# Orientation deltas
names = ["No filter", "Low pass filter"]
plt.plot(deltas_pure)
plt.plot(deltas_filtered)
plt.title("Orientation deltas [radians]")
plt.legend(names)
plt.show()

# Orientation
plt.plot(orients)
plt.title("Orientation [radians]")
plt.show()

# Position
plt.scatter(positions[:,0], positions[:,1],
            c = np.linspace(0, 10, len(positions)))
plt.plot(positions[:,0], positions[:,1], lw = 0.3, c = "red")
plt.text(0, 0, s = "Start")
plt.text(positions[-1,0], positions[-1,-1], s = "Finish")
plt.title("Robot position")
plt.colorbar(label="Time [s]")
plt.show()



