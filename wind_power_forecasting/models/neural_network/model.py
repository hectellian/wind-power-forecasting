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
import torch
from torch import nn

# Modules
from ..model import Model

class LSTM(Model):
    """Wrapper class for the neural network.
    
        Attributes
        ----------
        model : nn.Module
            The neural network.
        criterion : nn.L1Loss
            The loss function.
        optimizer : torch.optim.Optimizer
            The optimizer.
    """
    
    class Inner(nn.Module):
        """Neural network class.
    
        Attributes
        ----------
        lstm : nn.LSTM
            The LSTM layer.
        fc : nn.Linear
            The fully connected layer.
        hidden_size : int
            Number of hidden layers.
        layer_num : int
            Size of layers.

        Methods
        -------
        forward
            Forward pass.
        """
        def __init__(self, input_size: int, output_size: int, hidden_size:int, layer_num:int, device: str ='cpu') -> None:
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
            super(LSTM.Inner, self).__init__()
            
            self.device = device
            
            self.hidden_size = hidden_size
            self.layer_num = layer_num
            
            self.lstm = nn.LSTM(input_size, hidden_size, layer_num, batch_first=True)
            self.fc = nn.Linear(hidden_size, layer_num, output_size)
            
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
            h0 = torch.zeros(self.layer_num, x.size(0), self.hidden_size).requires_grad_().to(self.device)
            c0 = torch.zeros(self.layer_num, x.size(0), self.hidden_size).requires_grad_().to(self.device)
            
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
            out = self.fc(out[:, -1, :])
            
            return out
        
    def __init__(self, input_size: int, hidden_size, layer_num, output_size: int, lr:float = 0.05, device: str ='cpu') -> None:
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
        learning_rate : float
            Learning rate.
        """
        self.device = device
        self.model = self.Inner(input_size, output_size, hidden_size, layer_num, device=device).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)