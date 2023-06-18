#!/usr/bin/env python
""" Neural network module.

This module contains the neural network class that is used to predict the wind
power production of a wind farm.
"""

__authors__ = ["Anthony Christoforou", "Ethan Arm"]
__contact__ = ["anthony.christoforou@etu.unige.ch", "ethan.arm@etu.unige.ch"]
__copyright__ = "Copyright 2023, Unige"
__credits__ = ["Anthony Christoforou", "Ethan Arm"]
__date__ = "2023/05/06"
__deprecated__ = False
__license__ = "GPLv3"
__maintainer__ = "Anthony Christoforou"
__status__ = "Development"
__version__ = "0.0.1"

# Libraries
import os
import torch
from torch import nn
import pandas as pd

class WindLSTM(nn.Module):
    """Neural network class.
    
    Attributes
    ----------
    flatten : nn.Flatten
        The flatten layer.
    linear_relu_stack : nn.Sequential
        The sequential layer.

    Methods
    -------
    forward(x)
        Forward pass.
    """
    def __init__(self):
        """Constructs all the necessary attributes for the NeuralNetwork object.
        """
        super(WindLSTM, self).__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(14, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        
    def forward(self, x):
        """Forward pass.
        
        Returns
        -------
        logits : torch.Tensor
            The output tensor.
        """
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits