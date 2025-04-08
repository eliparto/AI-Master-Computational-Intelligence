""" Navigating model for targeted locomotion """

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

from sim_torch import Robot, genParams

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.ReLU()
            )
        self.softmax = nn.Softmax(dim = 0)
        
    def forward(self, x):
        x = self.stack(x)
        x = self.softmax(x)
        return x
    
def trainModel(model):
    velocities, startOrientation, loc_start, loc_target = genParams(bounds, k, sd)
    robot = Robot(velocities, startOrientation, loc_start, loc_target)
    outputs = []
    losses = []
    print(f"Velocities: {velocities}")
    
    model.train()
    for i in range(generations):
        optimizer.zero_grad()
        t_in = torch.tensor([
            torch.norm(robot.pos - robot.target_pos), robot.orient])
        w_out = model(t_in)
        robot.step(w_out)


        loss = torch.norm(robot.pos - robot.target_pos)
        loss.backward()
        optimizer.step()
        
        
        losses.append(loss.item()) 
        #outputs = torch.stack((outputs, w_out))
        
    return robot, losses, outputs
        
def abs_dist(t):
    """
    Return sqrt(x_t^2 + y_t^2).
    """
    return torch.sqrt(torch.sum(torch.square(t)))
        
# Setup
model = NN()
loss_func = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
generations = 2000

bounds = (-50, 50)
k = 2
sd = 1
