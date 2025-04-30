import numpy as np
import mujoco
from mujoco.viewer import launch_passive
from pyrr import Quaternion
import time 
import matplotlib.pyplot as plt

# from skopt import gp_minimize
# from skopt.space import Real
# from skopt.utils import use_named_args

from morphlib.brain._brain_cpg_instance import BrainCpgInstance
from morphlib.bodies.robogen.modules.active_joint import ActiveJoint

from morphlib.tools.build_file import build_mjcf
from morphlib.tools.mj_default_sim_setup import mujoco_setup_sim
from morphlib.terrains.mujoco_plane import mujoco_plane
from morphlib.brain._make_cpg_network_structure_neighbor import active_hinges_to_cpg_network_structure_neighbor

from morphlib.bodies.robogen.gecko_soft import gecko_soft
from morphlib.bodies.robogen.gecko import gecko
from main import get_actuator_ids, get_brain, objective_function


best_weights = [ 0.75049083,  0.81515644,  0.73604517,  0.30099464, -0.82977338,
        0.93550531,  0.98844977, -0.70604055,  0.05516819]

body = gecko()
actuator_ids = get_actuator_ids(body)

brain = get_brain(body, best_weights, actuator_ids)
# score = simulate_cpg_network(body, brain, render=True, num_steps=5_000)
objective_function([best_weights], actuator_ids, render=True)