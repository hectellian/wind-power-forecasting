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
from .models import *
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
    SEQ_LEN = 1
    INPUT_SIZE = 11
    HIDDEN_SIZE = 6
    OUTPUT_SIZE = 1
    NUM_LAYERS = 1
    BATCH_SIZE = 32
    LEARNING_RATE = 0.05
    EPOCHS = 500
    
    # Transforms
    transform = StandardScaler()
    target_transform = MinMaxScaler()
    
    # Load the dataset
    dataset = CustomWindFarmDataset(data_dir, relative_position_file, q=SEQ_LEN, device=device, transform=transform.fit_transform, target_transform=target_transform.fit_transform)
    #patv_correlations = dataset.correlations("Patv")
    #print("Correlations : ", patv_correlations)
    
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
    
    # Load the models
    nn_model = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, device=device, learning_rate=LEARNING_RATE, transform=transform.inverse_transform, target_transform=target_transform.inverse_transform)
    knn_model = KNN(device=device, transform=transform.inverse_transform, target_transform=target_transform.inverse_transform)

    # Print the dataset
    print(f"Dataset length: {len(dataset)}")

    train_sequence, train_target = next(iter(train_dataloader))
    print(f"Sequence batch shape: {train_sequence.size()}")
    print(f"Target batch shape: {train_target.size()}")

    # Print sequence and target
    print("First 5 sequences:")
    print(train_sequence[:5])
    print("First 5 targets:")
    print(train_target[:5])
    
    # Print models
    print(nn_model)
    print(knn_model)
    
    nn_model.load('./wind_power_forecasting/saved_models/nn_model.pt')
    with torch.no_grad():
        print("NN model prediction:")
        print(nn_model(train_sequence))
    
    # Neural Network Modeling
    nn_model.train(train_dataloader, validation_dataloader, epochs=EPOCHS, record_freq=10)
    nn_model.plot_loss()
    nn_model.plot_accuracy()
    nn_model.plot_prediction(test_dataloader)
    nn_model.save('./wind_power_forecasting/saved_models/nn_model.pt')
    
    # KNN Modeling    
    knn_model.train(train_dataloader, validation_dataloader)
    knn_model.plot_accuracy()
    knn_model.plot_prediction(test_dataloader)
    
if __name__ == "__main__":
    main()