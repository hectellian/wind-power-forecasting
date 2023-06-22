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

class KNN():
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

    def __init__(self, X = None, Y = None, k = 3) -> None:
        """Constructs the neccessary attributes and trains the model.

        Parameters
        ----------
        X: torch.Tensor
            The training points
        Y: torch.Tensor
            The training labels
        k: int
            The number of neighbors to be studied
        """
        self.train(X, Y)
        self.k = k

    def train(self, X, Y) -> None:
        """Trains the model over the given inputs.

        Parameters
        ----------
        X: torch.Tensor
            The training points
        Y: torch.Tensor
            The training labels
        """

        self.train_pts = X
        self.train_labels = Y

    def __call__(self, x):
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
        return self.predict(x)
    
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
        
        distances = torch.norm(self.train_pts-x,dim=1)
        knn = distances.topk(self.k, largest=False)
        
        neighbors = self.train_labels[knn.indices]

        return torch.mean(neighbors,dim=0)
