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
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Modules
from ..model import Model

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

    def __init__(self, k = 3, device = None, transform=None, target_transform=None) -> None:
        """Constructs the neccessary attributes and trains the model.

        Parameters
        ----------
        k: int
            The number of neighbors to be studied
        """
        self.k = k
        self.model = KNeighborsRegressor(n_neighbors=k, metric='manhattan')
        self.device = device
        self.transform = transform
        self.target_transform = target_transform

    def train(self, train_data:DataLoader, validation_data:DataLoader) -> None:
        """Trains the model over the given inputs.

        Parameters
        ----------
        X: DataLoader
            The training datas
        """
        features = []
        labels = []
        
        val_features = []
        val_labels = []
        
        for batch_points, batch_values in train_data:
            reshaped = torch.reshape(batch_points, (batch_points.shape[0] * batch_points.shape[1], batch_points.shape[2]))
            features.append(reshaped)
            labels.append(batch_values)
            
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        
        self.model.fit(features.cpu().numpy(), labels.cpu().numpy())
        
        for val_points, val_values in validation_data:
            reshaped_val = torch.reshape(val_points, (val_points.shape[0] * val_points.shape[1], val_points.shape[2]))
            val_features.append(reshaped_val)
            val_labels.append(val_values)
            
        val_features = torch.cat(val_features, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        
        predictions = self.predict(val_features)
        accuracy = np.sqrt(mean_squared_error(val_values, predictions))
        self.accuracy = accuracy
        
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
        return self.target_transform(torch.tensor(self.model.predict(x.cpu().numpy())))
    
    def plot_loss(self):
        return "Not plotabel since there is no loss function"
    
    def plot_prediction(self, test_data:DataLoader):
        X = torch.Tensor().to(self.device)
        y = torch.Tensor().to(self.device)

        for batch_points, batch_values in test_data:
            reshaped = torch.reshape(batch_points, (batch_points.shape[0] * batch_points.shape[1], batch_points.shape[2]))
            X = torch.cat((X, reshaped), dim=0)
            y = torch.cat((y, batch_values), dim=0)
             
        outputs = self.predict(X)
        y = self.target_transform(y.cpu().detach().numpy())
        plt.plot(outputs, label="Prediction Data")
        plt.plot(y, label="Real Data")
        plt.title("Active Power Prediction")
        plt.legend()
        plt.show()
        
    def plot_accuracy(self):
        plt.plot(self.accuracy, label="Accuracy")
        plt.title("Accuracy of Active Power")
        plt.legend()
        plt.show()
        
    def save(self, file_name:str):
        return f"Cannot save KNN model"

    def load(self, file_name:str):
        return f"Cannot load KNN model"

    def __str__(self) -> str:
        return f"KNN: - k: {self.k}"