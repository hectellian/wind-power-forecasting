#!/usr/bin/env python
""" Model module.

This module contains the Model class that is used as a Base Class
for each models subsequently defined.
"""

__authors__ = ["Anthony Christoforou", "Ethan Arm"]
__contact__ = ["anthony.christoforou@etu.unige.ch", "ethan.arm@etu.unige.ch"]
__copyright__ = "Copyright 2023, Unige"
__credits__ = ["Anthony Christoforou", "Ethan Arm"]
__date__ = "2023/06/23"
__deprecated__ = False
__license__ = "GPLv3"
__maintainer__ = "Ethan Arm"
__status__ = "Development"
__version__ = "0.0.1"

# Librairies
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class Model():


    class Inner(torch.nn.Module):
        pass


    def __init__(self, learning_rate = 0.01, transform=None, target_transform=None) -> None:
        
        self.model = self.Inner()
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.Optimizer()


    def predict(self, x:torch.Tensor):
        return self.target_transform(self.model(x).cpu().detach().numpy())

    def __call__(self, x):
        return self.predict(x)

    def train(self, train_data:"DataLoader", validation_data:"DataLoader", epochs = 200_000, record_freq=1_000):
        
        accuracy_history = []
        loss_record_list = []
        val_record_list = []
        intermediate_loss_list = []
        intermediate_val_list = []
        epoch_record_list:"list[int]" = []

        for epoch in tqdm(range(int(epochs)),desc=f"{self.__class__.__name__} Trainining"):

            for batch_points, batch_values in train_data:

                self.optimizer.zero_grad()
                outputs = self.model(batch_points)
                loss = self.criterion(outputs, batch_values)

                loss.backward()
                self.optimizer.step()

            if epoch%record_freq == 0:

                intermediate_loss_list.append(loss.item())
                epoch_record_list.append(epoch)

                predictions = []
                true_labels = []
                # TO not affect the computed weights
                with torch.no_grad():
                    for batch_points, batch_values in validation_data:

                        outputs = self.model(batch_points)
                        loss = self.criterion(outputs, batch_values)
                            
                        intermediate_val_list.append(loss.item())
                            
                        predictions.extend(outputs.cpu().numpy())
                        true_labels.extend(batch_values.cpu().numpy())

                accuracy = torch.tensor(mean_squared_error(true_labels, predictions, squared=False))
        
                loss_record_list.append(sum(intermediate_loss_list)/len(intermediate_loss_list))
                val_record_list.append(sum(intermediate_val_list)/len(intermediate_val_list))
                accuracy_history.append(accuracy)
        
        self.accuracy_record = accuracy_history
        self.loss_record = loss_record_list
        self.val_record = val_record_list
        self.epoch_record = epoch_record_list
        
        print(f"Training finished. Final Accuracy: {accuracy_history[-1]}")

        return loss_record_list.copy(), epoch_record_list.copy()
    

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: \n\t-model: {self.model} \n\t-criterion: {self.criterion} \n\t-optimizer: {self.optimizer}\n"

    def __repr__(self) -> str:
        return self.__str__()

    def save(self, file_name:str):
        print(f"Saving model to {file_name}.")
        torch.save(self.model, file_name)

    def load(self, file_name:str):
        print(f"Loading model from {file_name}.")
        torch.load(file_name)
        self.model.eval()
        
    def plot_loss(self):
        plt.plot(self.epoch_record,self.loss_record, label="Train Loss")
        plt.plot(self.epoch_record,self.val_record, label="Validation Loss")
        plt.title("Loss in function of training epochs")
        plt.legend()
        plt.show()
        
    def plot_prediction(self, test_data:"DataLoader"):
        predictions = []
        true_labels = []
        # TO not affect the computed weights
        with torch.no_grad():
            for batch_points, batch_values in test_data:
                outputs = self.model(batch_points)
                outputs = self.target_transform(outputs.cpu().numpy())
                predictions.extend(outputs)
                true_labels.extend(batch_values.cpu().numpy()) 
                
        true_labels = self.target_transform(true_labels)
        predictions = np.array(predictions)
        plt.plot(predictions, label="Prediction Data")
        plt.plot(true_labels, label="Real Data")
        plt.title("Active Power Prediction")
        plt.legend()
        plt.show()
        
    def plot_accuracy(self):    
        plt.plot(self.epoch_record, self.accuracy_record)
        plt.title("Accuracy of Active Power in function of training epochs")
        plt.show()
        
