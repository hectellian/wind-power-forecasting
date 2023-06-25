#!/usr/bin/env python
""" Random Forest module.

This module contains the RandomForest class that is used to predict the wind 
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

# Librairies
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna

# Modules
from ..model import Model

class RandomForest(Model):
    """ Random Forest class.

    The class is callable and calling it is equivalent to calling predict method.
    For more detail see the Model base class.

    """

    def __init__(self, n_trees = 100, max_deph = None, verbose = 0, device = None) -> None:
        """Constructs the neccessary attributes and trains the model.

        Parameters
        ----------
        n_trees: int = 100
            The number of trees in the fprest
        max_deph: int | None = None
            The maximum deph of the trees, if is None will progress till all leaves are pure.
        verbose: 0..10
            The degree of verbose to be shown in training, 10 being the max
        device = None
            The device to run on, see torch doc.
        """
        self.model = RandomForestRegressor(warm_start=True, n_estimators=n_trees, max_depth=max_deph)
        self.device = device

    def predict(self, x):
        return self.model.predict(x)

    def train(self, train_dataset, validation_dataset, batch_size=32, epochs=200, record_freq=10, trial=None):
        """  Trains the model according to the Datasets, and parameters. Returns the computed loss at the specied intervals.

        Parameters
        ----------
            train_dataset
                The dataset used to train the model

            validation_dataset
                The dataset uset to compute the metrics of the training

            batch_size: int = 32
                The size of a batch

            epochs: int = 200
                The number of epochs used in training

            record_freq: int = 10
                Every time epoch%record_freq == 0, the various metrics will be computed

            trial: None
                A parameter for optuna hyperparameter optimization
            
        Returns
        -------
            A copy of the computed metrics, which are also saved as an internal state
        """
        accuracy_history = []
        loss_record_list = []
        val_record_list = []
        epoch_record_list:"list[int]" = []
        
        # Create the dataloaders
        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_data = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

        for epoch in tqdm(range(int(epochs)),desc=f"{self.__class__.__name__} Trainining"):
            intermediate_loss_list = []
            for batch_points, batch_values in train_data:
                batch_points = np.squeeze(batch_points.cpu().numpy(), axis=1)
                batch_values = np.squeeze(batch_values.cpu().numpy())
                
                self.model.fit(batch_points,batch_values)
                ouptuts = self(batch_points)
                
                loss = mean_squared_error(batch_values, ouptuts)
                intermediate_loss_list.append(loss)

            if epoch%record_freq == 0:

                epoch_record_list.append(epoch)

                predictions = []
                true_labels = []
                intermediate_val_list = []

                for batch_points, batch_values in validation_data:
                    batch_points = np.squeeze(batch_points.cpu().numpy(), axis=1)
                    batch_values = np.squeeze(batch_values.cpu().numpy())

                    ouptuts = self(batch_points)
                    loss = mean_squared_error(batch_values,ouptuts)

                    intermediate_val_list.append(loss)

                    predictions.extend(ouptuts)
                    true_labels.extend(batch_values)
                
                accuracy = mean_absolute_error(true_labels, predictions)
        
                loss_record_list.append(sum(intermediate_loss_list)/len(intermediate_loss_list))
                val_loss = sum(intermediate_val_list)/len(intermediate_val_list)
                val_record_list.append(val_loss)
                accuracy_history.append(accuracy)
                
                if trial is not None:
                    trial.report(accuracy, epoch)      
                    if trial.should_prune():
                        raise optuna.TrialPruned()

        self.accuracy_record = accuracy_history
        self.loss_record = loss_record_list
        self.val_record = val_record_list
        self.epoch_record = epoch_record_list
        
        print(f"{self.__class__.__name__} Training finished. Final Accuracy: {accuracy_history[-1]}")
        
        return loss_record_list.copy(), val_record_list.copy()
        

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: \n\t model: {self.model}"
    
    def plot_prediction(self, test_dataset, batch_size=32, target_transform=None):
        """ Plot the prediction computed over the given dataset.
        """
        test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        true_labels = []
        # TO not affect the computed weights
        for batch_points, batch_values in test_data:
            batch_points = np.squeeze(batch_points.cpu().numpy(), axis=1)
            batch_values = np.squeeze(batch_values).reshape(-1, 1)
            outputs = self.model.predict(batch_points).reshape(-1, 1)
                
            if target_transform is not None:
                outputs = target_transform(outputs)
                    
            predictions.extend(outputs)
            true_labels.extend(batch_values.cpu().numpy())
        
        if target_transform is not None:        
            true_labels = target_transform(true_labels)
        predictions = np.array(predictions)
        plt.plot(predictions, label="Prediction Data")
        plt.plot(true_labels, label="Real Data")
        plt.title(f"{self.__class__.__name__} - Active Power Prediction")
        plt.legend()
        plt.show()