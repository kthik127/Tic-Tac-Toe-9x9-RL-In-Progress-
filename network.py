# building the actual neural network Q-function to calculate a Q(s,a) value given a board state
# using Conv2d to preserve board geometry, it's a Convolutional Neural Network (CNN)

import torch.nn as nn

class DQN(nn.Module): # new class representing our neural network Q-learning function
    def __init__(self):
        """
        in_channels: number of input channels (initially 4 from the board state per cell)
        out_channels: number of output channels
        kernel_size: 3 for a 3x3 sliding window for the board geometrically
        padding: controlling the spatial output size
        """
        super().__init__()
        # 3 convolutional layers for the 9x9 board total using sliding windows for learning / training
        self.conv1 = nn.Conv2d(4,64,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(128,128,kernel_size=3,padding=1)

        # ReLU layer
        # adds a nonlinear pattern between  layers so that they dont collapse, by using max function btwn the vector multiplication 
        # converts from simply linear to nonlinear due to max(0,x) function 

        self.relu = nn.ReLU() # f(x) = max(0,x), either shifting weights to 0 or 1, adding non-linear properties so that layers dont collapse

        # compression layers, from spatial to flat 
        self.fc1 = nn.Linear(10368, 512) # 128 x 9 x 9 = 10368 (compression)
        self.fc2 = nn.Linear(512, 81)

    def forward(self):
        """pass"""
