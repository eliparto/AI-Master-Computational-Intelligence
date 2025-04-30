import cma
import numpy as np
import mujoco
from mujoco.viewer import launch_passive
from pyrr import Quaternion
import time 
import matplotlib.pyplot as plt

from morphlib.brain._brain_cpg_instance import BrainCpgInstance
from morphlib.bodies.robogen.modules.active_joint import ActiveJoint

from morphlib.tools.build_file import build_mjcf
from morphlib.tools.mj_default_sim_setup import mujoco_setup_sim
from morphlib.terrains.mujoco_plane import mujoco_plane
from morphlib.brain._make_cpg_network_structure_neighbor import active_hinges_to_cpg_network_structure_neighbor

from morphlib.bodies.robogen.gecko_soft import gecko_soft
from morphlib.bodies.robogen.gecko import gecko
from morphlib.bodies.robogen.gecko_flippers import gecko_flippers

def get_actuator_ids(body):
    XML = build_mjcf(bodies=[body], body_poss = [[0,0,0]], body_oris = [Quaternion()], terrain_builder=mujoco_plane, sim_setup=mujoco_setup_sim, ts=0.001)
    ASSETS=dict()
    model = mujoco.MjModel.from_xml_string(XML, ASSETS)

    actuator_ids = []
    active_joints = body.find_modules_of_type(ActiveJoint)
    for actuator in active_joints:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator.name)
        actuator_ids.append(actuator_id)
    return actuator_ids


def get_brain(body, weights, actuator_ids):
    active_hinges = body.find_modules_of_type(ActiveJoint)
    cpg_network, _ = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
    weight_matrix = cpg_network.make_connection_weights_matrix_from_params(weights)
    
    # make initial state 
    initial_state = cpg_network.make_uniform_state(0.5 * np.sqrt(2))
    brain = BrainCpgInstance(initial_state, weight_matrix, actuator_ids)
    return brain


def objective_function_multi(solutions_mat, actuator_ids, render=False):
    body = gecko()
    brains = [
        get_brain(gecko(), solutions_mat[i].tolist(), actuator_ids) for i in range(3)
        ]
    score = simulate_cpg_network_multi(body, brains, render=render)
    
    return score
    

# Define the objective function
def objective_function(solutions, actuator_ids, render=False):
    result = []
    for weights in solutions:
        body = gecko()
        brain = get_brain(body, weights, actuator_ids)
        performance_score = simulate_cpg_network(body, brain, render=render)
        # print("Performance score: ", performance_score)
        result.append(performance_score)

    return np.asarray(result) # Minimize the negative performance score to maximize performance

def simulate_cpg_network_multi(body, brains, render=False, num_steps=1000, num_reps=1, dt=0.01):
    
    XML = build_mjcf(bodies=[body], body_poss = [[0,0,0]], body_oris = [Quaternion()], terrain_builder=mujoco_plane, sim_setup=mujoco_setup_sim, ts=dt)
    ASSETS=dict()
    scores = []
    
    model = mujoco.MjModel.from_xml_string(XML, ASSETS)
    data = mujoco.MjData(model)
    
    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "core")
    initial_core_pos = data.geom_xpos[core_id].copy()
    if render:
        with launch_passive(model, data) as viewer:
            # Close the viewer automatically after 10 wall-seconds.
            brain = brains[0]
            for i in range(num_steps):
                step_start = time.time()
                
                # Switch weights (dummy controller)
                if i == 333:
                    brain = brains[1]
                    print("Switching..")
                if i == 666: 
                    brain = brains[2]
                    print("Switching..")
                
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(model, data)
                brain.control(dt=dt, control_interface=data) # Control the robot
                # print(np.round(data.qpos,2))
                # data.ctrl[:] = 100
                # Pick up changes to the physics state, apply perturbations, update options from GUI.y
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    else:
        for _ in range(num_steps):
            mujoco.mj_step(model, data)
            # print(np.round(data.qpos,2))
            brain.control(dt=dt, control_interface=data) # Control the robot

    core_pos = data.geom_xpos[core_id].copy()
    initial_distance = initial_core_pos[0]
    distance = core_pos[0]

    score = -(distance - initial_distance)  # Minimize the negative distance to maximize performance
    scores.append(score)
    
    return np.mean(scores)

# Define the simulation function
def simulate_cpg_network(body, brain, render=False, num_steps=1000, num_reps=1, dt=0.01):
    
    XML = build_mjcf(bodies=[body], body_poss = [[0,0,0]], body_oris = [Quaternion()], terrain_builder=mujoco_plane, sim_setup=mujoco_setup_sim, ts=dt)
    ASSETS=dict()
    scores = []
    
    model = mujoco.MjModel.from_xml_string(XML, ASSETS)
    data = mujoco.MjData(model)
    
    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "core")
    initial_core_pos = data.geom_xpos[core_id].copy()
    if render:
        with launch_passive(model, data) as viewer:
            # Close the viewer automatically after 10 wall-seconds.
            for _ in range(num_steps):
                step_start = time.time()
                
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(model, data)
                brain.control(dt=dt, control_interface=data) # Control the robot
                # print(np.round(data.qpos,2))
                # data.ctrl[:] = 100
                # Pick up changes to the physics state, apply perturbations, update options from GUI.y
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    else:
        for _ in range(num_steps):
            mujoco.mj_step(model, data)
            # print(np.round(data.qpos,2))
            brain.control(dt=dt, control_interface=data) # Control the robot

    core_pos = data.geom_xpos[core_id].copy()
    initial_distance = initial_core_pos[0]
    distance = core_pos[0]

    score = -(distance - initial_distance)  # Minimize the negative distance to maximize performance
    scores.append(score)
    
    return np.mean(scores)



def main() -> None:
    NUM_GENERATIONS = 100
    initial_std = 0.5
    
    body = gecko()
    actuator_ids = get_actuator_ids(body)
    active_hinges = body.find_modules_of_type(ActiveJoint)
    cpg_network, _ = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
    
    initial_mean = cpg_network.num_connections * [0.5]

    options = cma.CMAOptions()
    options.set("bounds", [-1.0, 1.0])

    rng_seed = 12  # Cma seed must be smaller than 2**32.
    options.set("seed", rng_seed)
    opt = cma.CMAEvolutionStrategy(initial_mean, initial_std, options)

    generation_index = 0

    # Run cma for the defined number of generations.
    print("Start optimization process.")
    while generation_index < NUM_GENERATIONS:
        print(f"Generation {generation_index + 1} / {NUM_GENERATIONS}.")

        # Get the sampled solutions(parameters) from cma.
        solutions = opt.ask()

        # Evaluate them. Invert because fitness maximizes, but cma minimizes.
        fitnesses = objective_function(solutions, actuator_ids, render=False)

        # Tell cma the fitnesses.
        opt.tell(solutions, fitnesses)
        print(f"Fitnesses: {fitnesses}")
        # Increase the generation index counter.
        generation_index += 1
    
    print("Optimization process finished.")
    print(f"{opt.result.xbest=} {opt.result.fbest=}")


def test_weights(weights):
    body = gecko()
    actuator_ids = get_actuator_ids(body)

    brain = get_brain(body, weights, actuator_ids)
    score = simulate_cpg_network(body, brain, render=True, num_steps=1000, num_reps=1, dt=0.01)

    print(score)

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    print("Time taken: ", time.time() - start)
    # # weights = [-0.92213156,  0.69223841,  0.45479517, -0.97372374,  0.37796328, 0.38031153,  0.9162406 , -0.77868049, -0.8890298 ]
    # weights = [-0.960593  ,  0.96637397,  0.66855513, -0.96291011,  0.69724649, -0.56634243,  0.9998719 , -0.94532919, -0.68648703]
    # weights = [ 0.5254722 ,  0.50992647,  0.88913471, -0.28222639,  0.61965614,
    #     0.9973401 ,  0.96947946, -0.63274791,  0.05430416]
    # test_weights(weights)


