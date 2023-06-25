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
from sklearn.metrics import mean_absolute_error

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

    def __init__(self, k=3, device = None) -> None:
        """Constructs the neccessary attributes and trains the model.

        Parameters
        ----------
        k: int
            The number of neighbors to be studied
        """
        self.k = k
        self.model = KNeighborsRegressor(n_neighbors=k, metric='manhattan')
        self.device = device

    def train(self, train_dataset, validation_dataset) -> None:
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
        
        for i, j in zip(range(len(train_dataset)), range(len(validation_dataset))):
            points, values = train_dataset[i]
            val_points, val_values = validation_dataset[j]
            features.append(points.cpu())
            labels.append(values.cpu().numpy())
            val_features.append(val_points.cpu().numpy())
            val_labels.append(val_values.cpu().numpy())
            
        features = np.concatenate(features, axis=0)
        val_features = np.concatenate(val_features, axis=0)
        
        self.model.fit(features, labels)
        
        predictions = self.predict(val_features)
        accuracy = mean_absolute_error(val_labels, predictions)
        
        self.accuracy = accuracy
        
        print(f"KNN Training finished. Final Accuracy: {accuracy}")
        
        return accuracy
        
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
        return self.model.predict(x)
    
    def plot_loss(self):
        return "Not plotabel since there is no loss function"
    
    def plot_prediction(self, test_data, target_transform=None):  
        test_features = []
        test_labels = []
        for i in range(len(test_data)):
            test_points, test_values = test_data[i]  
            test_features.extend(test_points.cpu().numpy())
            test_labels.append(test_values.cpu().numpy())
            
        outputs = self.predict(test_features)
        if target_transform is not None:
            outputs = target_transform(outputs)
            test_labels = target_transform(test_labels)
        plt.plot(outputs, label="Prediction Data")
        plt.plot(test_labels, label="Real Data")
        plt.title("KNN - Active Power Prediction")
        plt.legend()
        plt.show()
        
    def plot_accuracy(self):
        return f"Accuracy: {self.accuracy}"
        
    def save(self, file_name:str):
        return f"Cannot save KNN model"

    def load(self, file_name:str):
        return f"Cannot load KNN model"

    def __str__(self) -> str:
        return f"KNN: - k: {self.k}"