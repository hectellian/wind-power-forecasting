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
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class Model():


    class Inner(torch.nn.Module):
        pass


    def __init__(self, learning_rate = 0.01) -> None:
        
        self.model = self.Inner()
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.Optimizer()


    def predict(self, x:torch.Tensor):
        return self.model(x)

    def __call__(self, x):
        return self.predict(x)

    def train(self, train_data:"DataLoader", epochs = 200_000, record_freq=1_000):
        
        loss_record_list = []
        epoch_record_list:"list[int]" = []

        for epoch in tqdm(range(int(epochs)),desc=f"{self.__class__.__name__} Trainining"):

            for batch_points, batch_values in train_data:

                self.optimizer.zero_grad()
                outputs = self.model(batch_points)
                loss = self.criterion(outputs, batch_values)

                loss.backward()
                self.optimizer.step()

            if epoch%record_freq == 0:

                epoch_record_list.append(epoch)

                # TO not affect the computed weights
                with torch.no_grad():

                    intermediate_loss_list = []

                    for batch_points, batch_values in train_data:

                        outputs = self.model(batch_points)
                        loss = self.criterion(outputs, batch_values)

                        intermediate_loss_list.append(loss.item())

                    loss_record_list.append(sum(intermediate_loss_list)/len(intermediate_loss_list))
        
        self.loss_record = loss_record_list
        self.epoch_record = epoch_record_list

        return loss_record_list.copy(), epoch_record_list.copy()
    

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: \n\t-model: {self.model} \n\t-criterion: {self.criterion} \n\t-optimizer: {self.optimizer}\n"

    def __repr__(self) -> str:
        return self.__str__()

    def save(self, file_name:str):
        torch.save(self.model.state_dict(), file_name)

    def load(self, file_name:str):
        self.model.load_state_dict(torch.load(file_name))
        
    def plot_loss(self):
        plt.plot(self.epoch_record,self.loss_record)
        plt.title("Loss in function of training epochs")
        plt.show()
        
    def plot_prediction(self, test_data:"DataLoader"):
        with torch.no_grad():
            prediction_list = []
            real_data_list = []
            for batch_points, batch_values in test_data:
                outputs = self.model(batch_points)
                for o, d in zip(outputs, batch_values):
                    prediction_list.append(o.item())
                    real_data_list.append(d.item())
            plt.plot(prediction_list, label="Prediction Data")
            plt.plot(real_data_list, label="Real Data")
            plt.title("Active Power Prediction")
            plt.legend()
            plt.show()
        
    def plot_accuracy(self, validation_data:"DataLoader"):    
        with torch.no_grad():
            accuracy_list = []
            for batch_points, batch_values in validation_data:
                outputs = self.model(batch_points)
                accuracy = 1 - abs(outputs - batch_values)/batch_values
                for a in accuracy:
                    accuracy_list.append(a.item())
            plt.plot(accuracy_list)
            plt.title("Accuracy of Active Power")
            plt.show()
        
