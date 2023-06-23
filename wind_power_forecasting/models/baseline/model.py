#!/usr/bin/env python
""" K Nearest Neighbhor module, also baseline.

This module contains the KNN class that is used to predict the wind 
power production of a wind farm.
"""

__authors__ = ["Anthony Christoforou", "Ethan Arm"]
__contact__ = ["anthony.christoforou@etu.unige.ch", "ethan.arm@etu.unige.ch"]
__copyright__ = "Copyright 2023, Unige"
__credits__ = ["Anthony Christoforou", "Ethan Arm"]
__date__ = "2023/06/22"
__deprecated__ = False
__license__ = "GPLv3"
__maintainer__ = "Anthony Christoforou"
__status__ = "Development"
__version__ = "0.0.1"

# Libraries
import torch
from ..model import Model
from torch.utils.data import DataLoader

class KNN(Model):
    """K Nearest Neighbor class.

    Attributes
    ----------
    k: int
        
    train_pts: Tensor
        The training points
    train_labels: Tensor
        The training labels / values

    Methods
    -------
    train(X, Y)
        The training of the model.
    predict(x)
        The prediction the label of x
    """

    def __init__(self, k = 3, device = None) -> None:
        """Constructs the neccessary attributes and trains the model.

        Parameters
        ----------
        k: int
            The number of neighbors to be studied
        """
        self.k = k
        self.device = device

    def train(self, train_data:DataLoader) -> None:
        """Trains the model over the given inputs.

        Parameters
        ----------
        X: DataLoader
            The training datas
        """

        self.data = train_data

    def predict(self, x):
        """Predict the label of x on the current model

        Parameters
        ----------
        x: torch.Tensor
            The studied tensor

        Returns
        -------
        label: torch.tensor
            The computed label for x
        """
        
        points = torch.Tensor(device=self.device)
        values = torch.Tensor(device=self.device)


        for batch_points, batch_values in self.data:
            points = torch.cat(points,batch_points)
            values = torch.cat(values,batch_values)

        distances = torch.norm(points-x,dim=1)
        knn = distances.topk(self.k, largest=False)
        
        neighbors = values[knn.indices]

        return torch.mean(neighbors,dim=0)
    
    def plot_loss(self):
        return "Not plotabel since there is no loss function"

    def __str__(self) -> str:
        return f"KNN: - k: {self.k}"