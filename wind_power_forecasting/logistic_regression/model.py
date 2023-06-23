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
from tqdm import tqdm

class LogisticRegression():

    class Inner(torch.nn.Module):

        def __init__(self, input_size, output_size ) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(input_size, output_size)

        def forward(self, x):
            outputs = self.linear(x)
            return outputs
        
    def __init__(self, X_train:torch.Tensor, Y_train:torch.Tensor, learning_rate = 0.01, epochs=200_000) -> None:
        
        self.training_points = X_train
        self.training_values = Y_train

        self.in_dim = X_train.shape()[-1]
        self.out_dim = Y_train.shape()[-1]
        self.lr = learning_rate
        self.epochs = epochs

        self.model = self.Inner(self.in_dim,self.out_dim)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr)
        

        
    def train(self):
        
        for epoch in tqdm(range(int(self.epochs)),desc="Logistic Regression Trainining"):
            
            self.optimizer.zero_grad()
            outputs = self.model(self.training_points)
            loss = self.criterion(torch.squeeze(outputs), self.training_values)

            loss.backward()
            self.optimizer.step()

    def predict(self, x):

        return self.model(x)

