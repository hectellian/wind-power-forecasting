#!/usr/bin/env python
""" Main module of the project.

This module is the main module of the project. It contains the main function
that is executed when the program is run. This will call and compare different
models to predict the wind power production of a wind farm.
"""

__authors__ = ["Anthony Christoforou", "Ethan Arm"]
__contact__ = ["anthony.christoforou@etu.unige.ch", "ethan.arm@etu.unige.ch"]
__copyright__ = "Copyright 2023, Unige"
__credits__ = ["Anthony Christoforou", "Ethan Arm"]
__date__ = "2023/05/06"
__deprecated__ = False
__license__ = "GPLv3"
__maintainer__ = "Anthony Christoforou"
__status__ = "Development"
__version__ = "0.0.1"

# Libraries
import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import urllib.request

# Modules
from .data import CustomWindFarmDataset
from .neural_network.model import WindLSTM
from .urls import data_url, relative_position_url, data_dir, relative_position_file

# Functions
def download_data(url: str, filename: str):
    """Download datasets from the internet if they are not already present.
    """
    
    data_dir = "./wind_power_forecasting/data/"
    
    if not os.path.exists(filename):
        if not os.path.exists(os.path.dirname(data_dir)):
            os.makedirs(os.path.dirname(data_dir))
            
        print(f"Downloading {filename}...")
        
        with tqdm(unit="B", unit_scale=True, miniters=1, desc=filename) as progress_bar:
            urllib.request.urlretrieve(url, filename=filename, reporthook=lambda block_num, block_size, total_size: progress_bar.update(block_num * block_size - progress_bar.n))
            
        print(f"{filename} downloaded.")
        
    else:
        print(f"{filename} already exists.")
    
def main():
    """ Main function of the project.
    """
    
    # Download the data if it is not already present
    download_data(data_url, data_dir)
    download_data(relative_position_url, relative_position_file)
    
    # Check if GPU is available
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using {device} device.")
    
    # hyperparameters
    INPUT_SIZE = 11
    HIDDEN_SIZE = 8
    OUTPUT_SIZE = 1
    NUM_LAYERS = 1
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 16
    EPOCHS = 1000
    
    # Transforms
    transform = StandardScaler().fit_transform
    target_transform = MinMaxScaler().fit_transform
    
    # Load the dataset
    dataset = CustomWindFarmDataset(data_dir, relative_position_file, device=device)
    patv_correlations = dataset.correlations("Patv")
    print("Correlations : ", patv_correlations)
    
    # split the data into train and test sets and validation sets
    split_size = 0.8 # 80% of the dataset for training
    split_frac = 0.5 # 50% of the remaining 20% of the dataset for validation
    train_size = int(len(dataset)*split_size)
    val_size = int((len(dataset) - train_size)*split_frac)
    
    # Created using indices from 0 to train_size.
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    
    # Created using indices from train_size to train_size + val_size.
    validation_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))

    # Created using indices from train_size to train_size + test_size.
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader, test_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False), DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load the Neural Network model
    nn_model = WindLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=LEARNING_RATE)

    # Print the dataset
    print(f"Train Dataset length: {len(train_dataset)}")

    train_sequence, train_target = next(iter(train_dataloader))
    print(f"Sequence batch shape: {train_sequence.size()}")
    print(f"Target batch shape: {train_target.size()}")

    # Print sequence and target
    print("First 5 sequences:")
    print(train_sequence[:5])
    print("First 5 targets:")
    print(train_target[:5])
    
    # Print model
    print(nn_model)
    
    # See the output of the model without training
    with torch.no_grad():
        print(f'Without training: {nn_model(train_sequence)}')
    
    # Train the model
    """ for epoch in range(EPOCHS):
        for sequence, target in train_dataloader:
            target = target.type(torch.LongTensor)
            sequence, target = sequence.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = nn_model(sequence)
            loss = criterion(output, target)
            
            loss.backward()
            
            optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, loss: {loss.item()}")
            
    # See the output of the model after training
    for sequence, target in test_dataloader:
        target = target.type(torch.LongTensor)
        sequence, target = sequence.to(device), target.to(device)
        with torch.no_grad():
            prediction = nn_model(sequence)
            data_prediction = prediction.cpu().numpy()
            dataY_plot = target.cpu().numpy()
            
            plt.figure(figsize=(20, 10))
            plt.axvline(x=200, c='r', linestyle='--') #size of the training set
            
            plt.plot(dataY_plot, label='Actuall Data') #actual plot
            plt.plot(data_prediction, label='Predicted Data') #predicted plot
            plt.title('Wind Power Time-Series Prediction')
            plt.legend()
            plt.show()  """
    
if __name__ == "__main__":
    main()