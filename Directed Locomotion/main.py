""" Directed Locomotion Simulator """

import numpy as np
from matplotlib import pyplot as plt

class robot(
        self, startLocation: list[list[float],float], velocitieslist[float]
        ) -> robot:
    def __init__(self) -> None:
        """
        :pos: Position of the robot
        :vect_vel: Vector containing forward and angular (left and right) velocities.
        """
        
        self.pos = startLocation[0]
        self.orient = startLocation[1]
        self.vect_vel = velocities
        
    def step(self, weights):
        """
        Perform a simulation step.
        """
        
        move = self.vect_vel * weights
        
        # Rotate body
        
def genParams() -> list[list[float],float]:
    pass
        

bounds = (-100.0, 100.0)
space = np.zeros(size = bound)

