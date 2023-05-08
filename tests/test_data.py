#!/usr/bin/env python
""" Test the WindDataset class.
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
from torchvision import transforms
import pandas as pd
import urllib.request
import pytest

# Modules
from wind_power_forecasting.data import WindDataset

class TestWindDataset:
    data_dir = "./wind_power_forecasting/data/wtbdata_245days.csv"
    relative_position_file = "./wind_power_forecasting/data/sdwpf_baidukddcup2022_turb_location.CSV"
    dataset = WindDataset(data_dir, relative_position_file)
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))

    def test_size(self):
        assert len(self.dataset) == 4678002

    def test_feature_size(self):
        # Number of features (14)
        assert self.train_features.shape[1] == 14
