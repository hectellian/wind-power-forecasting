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

class LSTM(nn.Module):
    """Neural network class.
    
    Attributes
    ----------
    flatten : nn.Flatten
        The flatten layer.
    linear_relu_stack : nn.Sequential
        The sequential layer.
    hidden_size : int
        Number of hidden layers.
    layer_num : int
        Size of layers.

    Methods
    -------
    forward(x)
        Forward pass.
    """
    def __init__(self, input_size: int, hidden_size: int, layer_num: int, output_size: int):
        """Constructs all the necessary attributes for the NeuralNetwork object.
        
        Parameters
        ----------
        input_size : int
            Number of input features.
        hidden_size : int
            Number of hidden layers.
        layer_num : int
            Size of layers.
        output_size : int
            Number of outputs.
        """
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        
        self.lstm = nn.LSTM(input_size, hidden_size, layer_num, batch_first=True) # batch_first=True causes input/output tensors to be of shape (batch_dim, seq_dim, input_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        out : torch.Tensor
            Predicted output.
        """
        h0 = torch.zeros(self.layer_num, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.layer_num, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        
        return out