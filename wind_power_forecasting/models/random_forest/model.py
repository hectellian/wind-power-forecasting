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
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

# Modules
from ..model import Model

class RandomForest(Model):

    def __init__(self, n_trees = 100, max_deph = None, verbose = 0, device = None) -> None:
        
        self.model = RandomForestRegressor(warm_start=True, n_estimators=n_trees, max_depth=max_deph)
        self.device = device

    def predict(self, x):
        return self.model.predict(x)


    def train(self, train_data: DataLoader, validation_data: DataLoader, epochs=200000, record_freq=1000):

        accuracy_history = []
        loss_record_list = []
        val_record_list = []
        intermediate_loss_list = []
        intermediate_val_list = []
        epoch_record_list:"list[int]" = []

        for epoch in tqdm(range(int(epochs)),desc=f"{self.__class__.__name__} Trainining"):

            for batch_points, batch_values in train_data:

                self.model.fit(batch_points.cpu().numpy(),batch_values.cpu().numpy)
                ouptuts = self(batch_points)
                loss = mean_squared_error(batch_values,ouptuts)


            if epoch%record_freq == 0:

                intermediate_loss_list.append(loss)
                epoch_record_list.append(epoch)

                predictions = []
                true_labels = []

                for batch_points, batch_values in validation_data:

                    ouptuts = self(batch_points)
                    loss = mean_squared_error(batch_values,ouptuts)

                    intermediate_val_list.append(loss)

                    predictions.extend(ouptuts)
                    true_labels.extend(batch_values)
                
                accuracy = mean_squared_error(true_labels, predictions, squared=False)
        
                loss_record_list.append(sum(intermediate_loss_list)/len(intermediate_loss_list))
                val_record_list.append(sum(intermediate_val_list)/len(intermediate_val_list))
                accuracy_history.append(accuracy)

        self.accuracy_record = accuracy_history
        self.loss_record = loss_record_list
        self.val_record = val_record_list
        self.epoch_record = epoch_record_list

        print(f"{self.__class__.__name__} Training finished. Final Accuracy: {accuracy_history[-1]}")
        

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: \n\t model: {self.model}"