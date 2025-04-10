""" Directed Locomotion Simulator 
TODO:   - Update casting types
        - Normalize positions wrt bounds
"""

import torch
import numpy as np
from matplotlib import pyplot as plt

class Robot():
    def __init__(self, velocities, target_pos):
        """
        :pos: Position of the robot
        :vect_vel: Vector containing forward and angular (left and right) velocities.
        """
        
        self.positions = []
        self.velocities = velocities
        self.target_pos = target_pos.tolist()
        
    def step(self, pos, alpha, weights):
        """
        Perform a simulation step.
        
        :weights: [w_forward, w_rot_left, w_rot_right]
        """
        
        # TODO: Detemine if masking the left/right rotational weights is useful
        actions = self.velocities * weights
        alpha = alpha + weights[1] + weights[2]
        disp = torch.stack((torch.cos(alpha), torch.sin(alpha))).view(2)
        pos = pos + disp
        
        # Position logging
        self.positions.append(pos.tolist())
        
        return pos, alpha
        
    def plotTrajectory(self) -> None:
        x, y = np.split(np.array(self.positions), 2, 1)
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        
        plt.scatter(x, y, c = np.linspace(0, 10, len(self.positions)))
        plt.scatter(self.target_pos[0], self.target_pos[1], c="red")
        plt.plot(x, y, lw = 0.1)
        #plt.quiver(x[:-1], y[:-1], dx, dy, width = 0.004, color = "gray",
                   #angles="xy")
        plt.text(x[0], y[0], s="Start")
        plt.text(x[-1], y[-1], s="Stop")
        plt.text(self.target_pos[0], self.target_pos[1], s="TGT")
        plt.colorbar(label="Time")
        plt.title("Robot trajectory")
        plt.show()
                
def genParams(bounds_space, sd_v, sd_w) -> list[list[float],float]:
    """
    Generate initial velocities and positions using numpy and convert to tensors.
    :return: [Velocity weights], orientation, robot position, target position.
    """
    # Initialize weights (forward and rotational velocities)
    v = np.abs(np.random.normal(loc=0, scale=sd_v, size = 1))
    w = np.abs(np.random.normal(loc=0, scale=sd_w, size = 2))
    v_vect = torch.tensor(np.hstack((v, w))).type(torch.float)
    
    # Initialize positions and orientation
    orient_robot = torch.tensor(np.random.uniform(low=-np.pi, high=np.pi),
                                requires_grad=True).type(torch.float).view(1)
    pos_robot = torch.tensor(np.random.uniform(bounds_space),
                             requires_grad=True).type(torch.float)
    pos_target = torch.tensor(np.random.uniform(bounds_space)).type(torch.float)
    
    return v_vect, orient_robot, pos_robot, pos_target

def genActions() -> list[float]:
    """
    PLACEHOLDER: Generate actions for simulated movement.
    """
    a_v = np.reshape(np.abs(np.random.normal(loc=0, scale=sd)))
    a_w = np.random.uniform(low = 0.0, high = 0.5, size = 2)
    #a_w = np.random.chisquare(df=k+2, size = 2)
    
    return np.hstack((a_v, a_w))

bounds_space = (-50.0, 50.0)
bounds_locs = bounds_space
k = 2
sd_v = 0.8
sd_w = 0.3
