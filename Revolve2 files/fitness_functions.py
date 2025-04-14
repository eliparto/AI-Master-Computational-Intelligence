"""Standard fitness functions for modular robots."""

import math
import numpy as np

from revolve2.modular_robot_simulation import ModularRobotSimulationState


def xy_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    """
    Calculate the distance traveled and orientation on the xy-plane by a single modular robot.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :returns: The calculated fitness abd natural orientation.
    """
    begin_pos = np.array([
        begin_state.get_pose().position.x, begin_state.get_pose().position.y])
    
    end_pos = np.array([
        end_state.get_pose().position.x, end_state.get_pose().position.y])
 
    # Positional displacement vector
    disp = end_pos - begin_pos
    
    # Robot's natural orientation
    beta = np.arctan2(disp[0], disp[1])
    
    return np.linalg.norm(disp), beta # Distance and orientation

def rotation(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    """
    Calculate the rotation about the core's axis by a single modular robot.
    This is done by converting the orientational quaternion' k component into 
    a rotation around the yaw (z) axis.
    """
    begin_orient = begin_state.get_pose().orientation
    end_orient = end_state.get_pose().orientation
    
    _, _, yaw_start = quaternion_to_euler(begin_orient)
    _, _, yaw_end = quaternion_to_euler(end_orient)
    
    return (yaw_end - yaw_start)
    
def quaternion_to_euler(q):
    w, x, y, z = q

    # X-axis rotation (roll)
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))

    # Y-axis rotation (pitch)
    pitch = np.arcsin(2*(w*y - z*x))

    # Z-axis rotation (yaw)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    return roll, pitch, yaw  # In radians