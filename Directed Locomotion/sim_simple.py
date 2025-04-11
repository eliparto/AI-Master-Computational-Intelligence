""" Directed Locomotion Simulator """

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

class Robot():
    def __init__(self, velocities: npt.NDArray[np.float_],
                 pos_robot: npt.NDArray[np.float_], 
                 pos_target: npt.NDArray[np.float_],
                 start_angle: float, threshold_angle: float) -> None:
        """
        :velocities: Vector containing forward and angular (left and right) velocities.
        :start_angle: World reference angle of robot.
        :pos_x: [x, y] position of x (robot or target).
        
        Class notation:
        :alpha: World reference angle to target (calculated through distance vector).
        :beta: World reference angle of robot.
        :action: {-1, 0, 1} -> Determines action (rotate/move forward).
        """
        
        self.pos_robot = pos_robot
        self.pos_target = pos_target
        self.beta = start_angle
        self.threshold = threshold_angle/360*2*np.pi # Converted to radians
        self.v = velocities[0]
        self.w_l = velocities[1]
        self.w_r = velocities[2]
        
        self.distance = self.pos_target - self.pos_robot
        self.alpha = np.arctan(self.distance[0] / self.distance[1])
        
        # Logging
        self.positions = np.array([self.pos_robot])
        self.orientations = np.array([self.beta])
        
    def step(self) -> None:
        # Compute angle to target safely
        self.distance = self.pos_target - self.pos_robot
        self.alpha = np.arctan2(self.distance[1], self.distance[0])
        
        # Angle difference accounting for wrapping
        angle_diff = (self.alpha - self.beta + np.pi) % (2 * np.pi) - np.pi
        
        # Rotate or move forward
        if np.abs(angle_diff) > self.threshold:
            if angle_diff > 0:
                self.beta += self.w_r
            else:
                self.beta -= self.w_l
        else:
            self.pos_robot += np.array([
                self.v * np.cos(self.beta),
                self.v * np.sin(self.beta)
            ])
        
        # Normalize angles
        self.beta = np.mod(self.beta, 2*np.pi)
        
        # Logging
        self.positions = np.vstack((self.positions, self.pos_robot))
        self.orientations = np.vstack((self.orientations, self.beta))
    
        
    def stepOld(self) -> None:
        """
        Determine the action to take depending on positions and angles.
        If angle beta > angle alpha -> rotate left, vice versa.
        
        If alpha > threshold -> rotate,
        Else -> Move forward
        """
        # Update angle beta
        if np.abs(self.alpha - self.beta) % (2 * np.pi) > self.threshold:
            if self.beta > self.alpha: self.beta -= self.w_l
            else: self.beta += self.w_r
         
        # Move forward
        else: 
            self.pos_robot += np.array(
                [self.v * np.cos(self.beta), self.v * np.sin(self.beta)])
        
        self.distance = self.pos_target - self.pos_robot
        self.alpha = np.arctan(self.distance[0] / self.distance[1])
        self.alpha = np.mod(self.alpha, 2*np.pi)
        self.beta = np.mod(self.beta, 2*np.pi)
            
        self.positions = np.vstack((self.positions, self.pos_robot))
        self.orientations = np.vstack((self.orientations, self.beta))
        
    def plotTrajectory(self) -> None:
        x, y = np.split(np.array(self.positions), 2, 1)
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        
        plt.scatter(x, y, c = np.linspace(0, 10, len(self.positions)))
        plt.scatter(self.pos_target[0], self.pos_target[1], c="red")
        plt.plot(x, y, lw = 0.1)
        #plt.quiver(x[:-1], y[:-1], dx, dy, width = 0.004, color = "gray",
                   #angles="xy")
        plt.text(x[0], y[0], s="Start")
        plt.text(x[-1], y[-1], s="Stop")
        plt.text(self.pos_target[0], self.pos_target[1], s="TGT")
        plt.colorbar(label="Time")
        plt.title("Robot trajectory")
        plt.show()
                
def genParams(bounds_space, sd_v, sd_w) -> list[list[float],float]:
    """
    Generate initial velocities and positions.
    :return: [Velocity weights], orientation, robot position, target position.
    """
    # Initialize weights (forward and rotational velocities)
    v = np.abs(np.random.normal(loc=0, scale=sd_v, size = 1))
    w = np.abs(np.random.normal(loc=0, scale=sd_w, size = 2))
    
    # Initialize positions
    orient_robot = np.random.uniform(low=-np.pi, high=np.pi)
    pos_robot = np.random.uniform(bounds_space)
    pos_target = np.random.uniform(bounds_space)
    
    return np.hstack((v, w)), pos_robot, pos_target, orient_robot

def run(dist) -> None:
    vel, pos_r, pos_t, alpha_start = genParams(bounds_space, sd_v, sd_w)
    robot = Robot(vel, pos_r, pos_t, alpha_start, alpha_t)
    
    while np.linalg.norm(robot.distance) > dist:
        robot.step()
        
    robot.plotTrajectory()

bounds_space = (-50.0, 50.0)
bounds_locs = bounds_space
sd_v = 0.5
sd_w = 0.1
alpha_t = 5


