""" Directed Locomotion Simulator """

import numpy as np
from matplotlib import pyplot as plt

class Robot():
    def __init__(self, velocitiesList: list[float], startOrientation: float,
                 startLocation: list[float], targetLocation: list[float]):
        """
        :pos: Position of the robot
        :vect_vel: Vector containing forward and angular (left and right) velocities.
        """
        
        self.pos = startLocation
        self.orient = startOrientation
        self.vect_vel = velocitiesList
        self.positions = np.array(self.pos)
        self.orientations = [self.orient]
        self.target_pos = targetLocation
        
    def step(self, weights) -> None:
        """
        Perform a simulation step.
        
        :weights: [w_forward, w_rot_left, w_rot_right]
        """
        
        # Update body rotation and position
        v, w_l, w_r = self.vect_vel * weights
        
        # Sum weight for rotation action
        if w_l > w_r: 
            w_l += w_r
            w_l = -w_l
            w_r = 0
        else:
            w_r += w_l
            w_l = 0
            
        self.orient += (w_l + w_r)
        #if self.orient < 0: self.orient = -np.mod(self.orient, np.pi)
        #else: self.orient = np.mod(self.orient, np.pi)
            
        disp_x = v * np.cos(self.orient)
        disp_y = v * np.sin(self.orient)
        pos_x, pos_y = self.pos
        pos_x += disp_x
        pos_y += disp_y
        self.pos = np.array([pos_x, pos_y])
        
        self.positions = np.vstack((self.positions, self.pos))
        self.orientations.append(self.orient)
        
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
        
    def distance(self) -> float:
        dist = np.power((self.target_pos - self.pos), 2)
        return np.sqrt(dist[0] + dist[1])
                
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
    a_w = np.random.chisquare(df=k+2, size = 2)
    
    return np.hstack((a_v, a_w))

bounds_space = (-50.0, 50.0)
bounds_locs = bounds_space
k = 2
sd = 1

# for _ in range(50):
#     actions = genActions()
#     robot.step(actions)
    
# plotTrajectory(positions, init_pos_t)


