""" Directed Locomotion Simulator """

import numpy as np
from matplotlib import pyplot as plt

class Robot():
    def __init__(self, velocitiesList: list[float], startOrientation: float,
                 startLocation: list[float]):
        """
        :pos: Position of the robot
        :vect_vel: Vector containing forward and angular (left and right) velocities.
        """
        
        self.pos = startLocation
        self.orient = startLocation[1]
        self.vect_vel = velocitiesList
        self.positions = []
        
    def step(self, weights) -> None:
        """
        Perform a simulation step.
        
        :weights: [w_forward, w_rot_left, w_rot_right]
        """
        
        # Sum weight for rotation action
        if weights[1] > weights[2]: 
            weights[1] += weights[2]
            weights[1] = -weights[1]
            weights[2] = 0
        else:
            weights[2] += weights[1]
            weights[1] = 0
            
        # Update body rotation and position
        v, w_l, w_r = self.vect_vel * weights
        self.orient += (w_l + w_r)
        vel_x = v * np.cos(self.orient)
        vel_y = v * np.sin(self.orient)
        pos_x, pos_y = self.pos
        pos_x += vel_x
        pos_y += vel_y
        self.pos = [pos_x, pos_y]
        self.positions.append(self.pos)
        
    def returnPos(self) -> list[float]:
        """
        Return the robot's positions.
        """
        return np.array(self.positions)
                
def genParams(bounds_space, df, sd_v) -> list[list[float],float]:
    """
    Generate initial velocities and positions.
    :return: [Velocity weights], orientation, robot position, target position.
    """
    # Initialize weights
    v = np.reshape(np.abs(np.random.normal(loc=0, scale=sd_v)), 1)
    w = np.random.chisquare(df=df, size = 2)
    
    # Initialize positions
    orient_robot = np.random.uniform(low=0.0, high=np.pi*2)
    pos_robot = np.random.uniform(bounds_space)
    pos_target = np.random.uniform(bounds_space)
    
    return np.hstack((v, w)), orient_robot, pos_robot, pos_target

def genActions() -> list[float]:
    """
    PLACEHOLDER: Generate actions for simulated movement.
    """
    a_v = np.reshape(np.abs(np.random.normal(loc=0, scale=sd*2)), 1)
    a_w = np.random.chisquare(df=k+2, size = 2)
    
    return np.hstack((a_v, a_w))

def plotTrajectory(positions, target) -> None:
    x, y = np.split(positions, 2, 1)
    plt.scatter(x, y, c = np.linspace(0, 10, len(positions)))
    plt.scatter(target[0], target[1], c="red")
    plt.plot(x, y, lw = 0.1)
    plt.text(x[0], y[0], s="Start")
    plt.text(x[-1], y[-1], s="Stop")
    plt.text(target[0], target[1], s="TGT")
    plt.title("Robot trajectory")
    plt.colorbar(label="Time")
    plt.show()

bounds_space = (-50.0, 50.0)
bounds_locs = bounds_space
k = 2
sd = 1

init_weights, init_orient, init_pos_r, init_pos_t = genParams(
    bounds_space, k, sd)
robot = Robot(init_weights, init_orient, init_pos_r)

for _ in range(50):
    actions = genActions()
    robot.step(actions)
    
plotTrajectory(positions, init_pos_t)


