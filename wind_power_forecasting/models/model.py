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
import optuna
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt

class Model():

    class Inner(torch.nn.Module):
        pass


    def __init__(self) -> None:
        
        self.model = self.Inner()
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.Optimizer()

    def predict(self, x:torch.Tensor):
        return self.model(x).to(x.device)
        
    def __call__(self, x):
        return self.predict(x)

    def train(self, trial, train_dataset, validation_dataset, batch_size=32, epochs=200, record_freq=10):
        accuracy_history = []
        loss_record_list = []
        val_record_list = []
        
        epoch_record_list:"list[int]" = []
        
        # Create the dataloaders
        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_data = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

        for epoch in tqdm(range(int(epochs)),desc=f"{self.__class__.__name__} Training"):
            intermediate_loss_list = []
            for batch_points, batch_values in train_data:

                self.optimizer.zero_grad()
                outputs = self.model(batch_points)
                loss = self.criterion(outputs, batch_values)

                intermediate_loss_list.append(loss.item())
                
                loss.backward()
                self.optimizer.step()

            if epoch%record_freq == 0:

                epoch_record_list.append(epoch)      
                loss_record_list.append(sum(intermediate_loss_list)/len(intermediate_loss_list))

                true_labels = []
                predicted_labels = []
                intermediate_val_list = []
                # TO not affect the computed weights
                with torch.no_grad():
                    for batch_points, batch_values in validation_data:

                        outputs = self.model(batch_points)
                        loss = self.criterion(outputs, batch_values)
                        
                        intermediate_val_list.append(loss.item())
                        
                        true_labels.extend(batch_values.cpu().numpy())
                        predicted_labels.extend(outputs.cpu().numpy())
            
                    accuracy = mean_absolute_error(true_labels, predicted_labels)
                    val_loss = sum(intermediate_val_list)/len(intermediate_val_list)
                    val_record_list.append(val_loss)
                    accuracy_history.append(accuracy)
                    
                    trial.report(val_loss, epoch)
                    
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            
        print(f"{self.__class__.__name__} Training finished. Final Accuracy: {accuracy}")
        
        self.accuracy_history = accuracy_history
        self.loss_record = loss_record_list
        self.val_record = val_record_list
        self.epoch_record = epoch_record_list

        return loss_record_list.copy(), val_record_list.copy()
    

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: \n\t-model: {self.model} \n\t-criterion: {self.criterion} \n\t-optimizer: {self.optimizer}\n"

    def __repr__(self) -> str:
        return self.__str__()

    def save(self, file_name:str):
        print(f"Saving self to {file_name}.")
        torch.save(self, file_name)

    def load(file_name:str):
        print(f"Loading objet from {file_name}.")
        return torch.load(file_name)
        
    def plot_loss(self):
        plt.plot(self.epoch_record,self.loss_record, label="Train Loss")
        plt.plot(self.epoch_record,self.val_record, label="Validation Loss")
        plt.title(f"{self.__class__.__name__} - Loss in function of training epochs")
        plt.legend()
        plt.show()
        
    def plot_prediction(self, test_dataset, batch_size=32, target_transform=None):
        test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        true_labels = []
        # TO not affect the computed weights
        with torch.no_grad():
            for batch_points, batch_values in test_data:
                outputs = self.model(batch_points).cpu().numpy()
                
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
        
    def plot_accuracy(self):    
        plt.plot(self.epoch_record,self.accuracy_history, label="Accuracy")
        plt.title(f"{self.__class__.__name__} - Accuracy in function of training epochs")
        plt.legend()
        plt.show()
        
