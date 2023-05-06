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
from torch.utils.data import DataLoader
import pandas as pd
import urllib.request

# Modules
from .data import WindDataset
from .utils.progressbar import ProgressBar

# Functions
def download_data(url, filename):
    """Download datasets from the internet if they are not already present.
    """
    
    data_dir = "./wind_power_forecasting/data/"
    
    if not os.path.exists(filename):
        if not os.path.exists(os.path.dirname(data_dir)):
            os.makedirs(os.path.dirname(data_dir))
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename, ProgressBar())
        print(f"{filename} downloaded.")
    else:
        print(f"{filename} already exists.")

def main():
    """ Main function of the project.
    """
    
    # Download the data if it is not already present
    data_url = "https://bj.bcebos.com/v1/ai-studio-online/85b5cb4eea5a4f259766f42a448e2c04a7499c43e1ae4cc28fbdee8e087e2385?responseContentDisposition=attachment%3B%20filename%3Dwtbdata_245days.csv&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-05-05T14%3A17%3A03Z%2F-1%2F%2F5932bfb6aa3af1bcfb467bf2a4a6877f8823fe96c6f4fd0d4a3caa722354e3ac"
    relative_position_url = "https://bj.bcebos.com/v1/ai-studio-online/e927ce742c884955bf2a667929d36b2ef41c572cd6e245fa86257ecc2f7be7bc?responseContentDisposition=attachment%3B%20filename%3Dsdwpf_baidukddcup2022_turb_location.CSV&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-04-11T08%3A27%3A09Z%2F-1%2F%2Fcf377452dbd186873680f2f0fe39200b3de86083a036da220ab8a02abc5a8032"
    
    data_dir = "./wind_power_forecasting/data/wtbdata_245days.csv"
    relative_position_file = "./wind_power_forecasting/data/sdwpf_baidukddcup2022_turb_location.CSV"
    
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
    
    # Load the dataset
    dataset = WindDataset(data_dir, relative_position_file)
    # We take the first 100 samples just to test the code
    dataset = dataset[:100]
    
    dataloader = DataLoader(dataset, batch_size=20, shuffle=False)
    
    print(dataset)
    print(f"Dataset length: {len(dataset)}")
    
    
if __name__ == "__main__":
    main()