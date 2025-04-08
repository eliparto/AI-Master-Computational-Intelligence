""" Directed Locomotion Simulator 
TODO: Update casting types
"""

import torch
import numpy as np
from matplotlib import pyplot as plt

class Robot():
    def __init__(self, velocitiesList: list[float], startOrientation: float,
                 startLocation: list[float], targetLocation: list[float]):
        """
        :pos: Position of the robot
        :vect_vel: Vector containing forward and angular (left and right) velocities.
        """
        
        self.pos = torch.tensor(startLocation).type(torch.float)
        self.orient = torch.tensor(startOrientation).type(torch.float)
        self.vect_vel = torch.tensor(velocitiesList).type(torch.float)
        self.positions = np.array([np.array(self.pos.detach())])
        self.orientations = [self.orient]
        self.target_pos = torch.tensor(targetLocation).type(torch.float)
        
    def step(self, weights) -> None:
        """
        Perform a simulation step.
        
        :weights: [w_forward, w_rot_left, w_rot_right]
        """
        
        # Update body rotation and position
        # TODO: See if network learns to disable counterrotation
        actions = self.vect_vel * weights
        # if actions[1] > actions[2]:
        #     actions[1] += actions[2]
        #     actions[2] = 0
        # else:
        #     actions[2] += actions[1]
        #     actions[1] = 0
            
        self.orient = self.orient + (actions[1] + actions[2])
        
        # # Update position using displacement
        disp = torch.stack([
            actions[0] * torch.cos(self.orient),
            actions[0] * torch.sin(self.orient)])
        self.pos = self.pos + disp
        self.positions = np.vstack((self.positions, np.array(self.pos.detach())))
        
    def plotTrajectory(self) -> None:
        x, y = np.split(np.array(self.positions), 2, 1)
        target = self.target_pos.detach()
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        
        plt.scatter(x, y, c = np.linspace(0, 10, len(self.positions)))
        plt.scatter(target[0], target[1], c="red")
        plt.plot(x, y, lw = 0.1)
        #plt.quiver(x[:-1], y[:-1], dx, dy, width = 0.004, color = "gray",
                   #angles="xy")
        plt.text(x[0], y[0], s="Start")
        plt.text(x[-1], y[-1], s="Stop")
        plt.text(target[0], target[1], s="TGT")
        plt.colorbar(label="Time")
        plt.title("Robot trajectory")
        plt.show()
                
def genParams(bounds_space, df, sd_v) -> list[list[float],float]:
    """
    Generate initial velocities and positions.
    :return: [Velocity weights], orientation, robot position, target position.
    """
    # Initialize weights (forward and rotational velocities)
    v = np.reshape(np.abs(np.random.normal(loc=0, scale=sd_v)), 1)
    w = np.random.chisquare(df=df, size = 2)
    
    # Initialize positions
    orient_robot = np.random.uniform(low=-np.pi, high=np.pi)
    pos_robot = np.random.uniform(bounds_space)
    pos_target = np.random.uniform(bounds_space)
    
    return np.hstack((v, w)), orient_robot, pos_robot, pos_target

def genActions() -> list[float]:
    """
    PLACEHOLDER: Generate actions for simulated movement.
    """
    a_v = np.reshape(np.abs(np.random.normal(loc=0, scale=sd*2)), 1)
    a_w = np.random.uniform(low = 0.0, high = 0.5, size = 2)
    #a_w = np.random.chisquare(df=k+2, size = 2)
    
    return np.hstack((a_v, a_w))

bounds_space = (-50.0, 50.0)
bounds_locs = bounds_space
k = 2
sd = 1

