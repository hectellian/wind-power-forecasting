#!/usr/bin/env python
""" Logistic Regression module.

This module contains the logistic regression class that is used to predict the wind
power production of a wind farm.
"""

__authors__ = ["Anthony Christoforou", "Ethan Arm"]
__contact__ = ["anthony.christoforou@etu.unige.ch", "ethan.arm@etu.unige.ch"]
__copyright__ = "Copyright 2023, Unige"
__credits__ = ["Anthony Christoforou", "Ethan Arm"]
__date__ = "2023/06/22"
__deprecated__ = False
__license__ = "GPLv3"
__maintainer__ = "Ethan Arm"
__status__ = "Development"
__version__ = "0.0.1"

# Libraries
import torch
from ..model import Model

class LogisticRegression(Model):
    """ The Logistic regression Model

    See documentation of the Model Base Class.
    """

    class Inner(torch.nn.Module):

        def __init__(self, input_size, output_size, device) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(input_size, output_size, device=device)

        def forward(self, x):
            outputs = torch.sigmoid(self.linear(x[:, -1, :]))
            return outputs
        
    def __init__(self, in_dim: int, out_dim:int, lr = 0.05, device = None) -> None:
        """Constructs the neccessary attributes and trains the model.

        Parameters
        ----------
        in_dim: int
            The input features dimension
        out_dim: int
            The label dimensions
        lr: float
            The learning rate of the model
        device = None
            The device where the computation are done, see torch doc.
        """
        self.device = device
        self.model = self.Inner(in_dim,out_dim, device=self.device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr)

