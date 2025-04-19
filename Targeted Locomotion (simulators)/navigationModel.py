""" Navigating model for targeted locomotion 
TODO: - Normalize inputs (positions as ratio between bounds)
"""

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

from sim_torch_position import Robot, genParams

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(3, 32),     # Input: [x, y, alpha]
            nn.Tanh(),            # Smoother transitions
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),     # Output: [forward, left, right]
            nn.Sigmoid()          # To scale velocities ∈ [0, 1]
        )
        
    def forward(self, x):
        x = self.stack(x)
        return x
    
def trainModel(model):
    """
    Initialize a robot and train it.
    :input_tensor: [pos_x, pos_y, alpha].
    :actionWeights: Vect w containing the multiplication factors for the velocity vector [v, w_l, w_r].
    """
    velocities, alpha_start, pos_start, pos_target = genParams(bounds, sd_v, sd_w)
    robot = Robot(velocities, pos_target)
    losses = []
    w_out = []
    print(f"Robot velocities: {velocities}")
    
    model.train()
    pos_robot = pos_start
    alpha = alpha_start
    
    # Normalize position and angle
    pos_robot = pos_robot / 50
    alpha = alpha / tau
    
    input_tensor = torch.concatenate((pos_robot, alpha_start))
    for i in range(generations):
        action_weights = model(input_tensor)
        pos_robot, alpha = robot.step(pos_robot, alpha, action_weights)
        
        loss = torch.norm((pos_robot - pos_target))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Input for the next step
        pos_robot = pos_robot / 50
        alpha = alpha / tau
        pos_robot = pos_robot.detach()
        alpha = alpha.detach()
        input_tensor = torch.concatenate((pos_robot, alpha))
        w_out.append(action_weights.tolist())
        losses.append(loss.item())
        
    plt.plot(losses)
    plt.title("Loss")
    plt.show()
    robot.plotTrajectory()
        
    return robot, losses, w_out
        
# Setup
bounds = (-50, 50)
k = 2
tau = np.pi * 2
sd_v = 0.8
sd_w = 0.5

model = NN()
loss_func = None
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
generations = 1000
