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
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
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
    INPUT_SIZE = 12
    HIDDEN_SIZE = 50
    OUTPUT_SIZE = 1
    NUM_LAYERS = 1
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 16
    EPOCHS = [5, 10, 50, 100, 200, 500, 1000]
    
    # Load the dataset
    dataset = CustomWindFarmDataset(data_dir, relative_position_file, device=device)
    patv_correlations = dataset.correlations("Patv")
    print("Correlations : ", patv_correlations)
    
    # Created using indices from 0 to train_size.
    train_dataset = torch.utils.data.Subset(dataset, range(int(len(dataset)*0.8)))

    # Created using indices from train_size to train_size + test_size.
    test_dataset = torch.utils.data.Subset(dataset, range(int(len(dataset)*0.8), int(len(dataset)*0.8) + (len(dataset) - int(len(dataset)*0.8))))

    # Create the dataloaders
    train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Load the model
    nn_model = WindLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)

    # Print the dataset
    print(f"Train Dataset length: {len(train_dataset)}")

    train_sequence, train_target = next(iter(train_dataloader))
    print(f"Sequence batch shape: {train_sequence.size()}")
    print(f"Target batch shape: {train_target.size()}")

    # Print sequence and target
    print("First 5 sequences:")
    print(train_sequence[:20])
    print("First 5 targets:")
    print(train_target[:20])
    
    # Print model
    print(nn_model)
    
if __name__ == "__main__":
    main()