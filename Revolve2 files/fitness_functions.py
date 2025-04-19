"""Standard fitness functions for modular robots."""

import numpy as np
from revolve2.modular_robot_simulation import ModularRobotSimulationState

class FitnessEvaluator():
    """ Calculate fitnesses based on a robot's simulation scenes. """
    
    def __init__(
            self, states: list[ModularRobotSimulationState]) -> None:
        self.states = states # List of simulation scene objects
        
        # Fitness weights
        self.p_forward = 1.0
        self.p_rotate = 1.0
        
    def xy_displacement(self) -> float:
        """
        Calculate the distance traveled and orientation on the xy-plane by a single modular robot.
        """
        # (x, y) Start and end position vectors
        begin_pos = np.array([
            self.states[0].get_pose().position.x,
            self.states[0].get_pose().position.y])
        end_pos = np.array([
            self.states[-1].get_pose().position.x, 
            self.states[-1].get_pose().position.y
            ])
     
        # Positional displacement vector
        disp = end_pos - begin_pos
        fit_disp = np.linalg.norm(disp)
        
        # Robot's natural orientation
        beta = np.arctan2(disp[0], disp[1])
        
        return fit_disp, beta # Distance and orientation

    def rotation(self) -> float:
        """
        Calculate the rotation about the core's axis by a single modular robot.
        This is done by converting the orientational quaternion' k component into 
        a rotation around the yaw (z) axis for every time delta
        """
        fit_rot = 0.0
        deltas_pure = []
        deltas_filtered = []
        orients = []
        
        for i in range(len(self.states)-1):
            state_t = self.states[i]       # State at time t
            state_t_1 = self.states[i+1]   # State at time (t+1)
            begin_orient = state_t.get_pose().orientation
            end_orient = state_t_1.get_pose().orientation
            _, _, yaw_start = self.quaternion_to_euler(begin_orient)
            _, _, yaw_end = self.quaternion_to_euler(end_orient)
            
            delta = yaw_end - yaw_start 
            deltas_pure.append(delta)
            # Low pass filter
            if abs(delta) > np.pi: delta = 0 # Prevent modulo operation from blowing up the delta
            fit_rot += delta
            deltas_filtered.append(delta)
            orients.append(yaw_end)
        
        return fit_rot, deltas_pure, deltas_filtered, orients
    
    def displacement(self):
        positions = np.zeros(2)
        for state in self.states:
            pos = np.array([
                state.get_pose().position.x,
                state.get_pose().position.y
                ])
            positions = np.vstack((positions, pos))
            
        return positions
        
    def quaternion_to_euler(self, q) -> tuple[float]:
        """
        Convert quaterion data into angles about roll, pitch and yaw axes.
        """
        w, x, y, z = q
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    
        return roll, pitch, yaw  # Angles in radians
    