#!/usr/bin/env python
""" Custom dataset class.

This module contains the custom dataset class that will be used to load the
wind power production data for the different models.
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
import torch
import pandas as pd

class CustomWindFarmDataset(torch.utils.data.Dataset):
    """A class to load the wind power production data.

    Attributes
    ----------
    data : pandas.DataFrame
        The data loaded from the data file.
    relative_positions : pandas.DataFrame
        The relative positions loaded from the relative position file.
    transform : callable, optional
        A function/transform that takes in a sample and returns a transformed
        version.
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.

    Methods
    -------
    __len__()
        Returns the length of the dataset.
    __getitem__(index)
        Returns one sample of the dataset.
    """
    def __init__(self, data_dir, relative_position_file, transform=None, target_transform=None):
        """Constructs all the necessary attributes for the Dataset object.

        Parameters
        ----------
        data_dir : str
            The path to the data file.
        relative_position_file : str
            The path to the relative position file.
        transform : callable, optional
            A function/transform that takes in a sample and returns a transformed
            version.
        target_transform : callable, optional
            A function/transform that takes in the target and transforms it.
        """
        self.data = pd.read_csv(data_dir)
        self.relative_positions = pd.read_csv(relative_position_file)
        self.transform = transform
        self.target_transform = target_transform

        # Merge the data and relative positions
        self.merged_data = pd.merge(self.data, self.relative_positions, on='TurbID')

        # Convert days to continuous minutes
        self.merged_data["Tmstamp"] = pd.to_datetime(self.merged_data["Tmstamp"], format='%H:%M')
        self.merged_data["Tmstamp"] = self.merged_data["Tmstamp"].dt.hour * 60 + self.merged_data["Tmstamp"].dt.minute

        # Handle missing values
        self.merged_data = self.merged_data.dropna() # Remove rows with missing values
        self.merged_data.iloc[:, -5:] = self.merged_data.iloc[:, -5:].clip(lower=0) # Replace negative values with 0
    
    def __len__(self):
        """Returns the number of samples.
        """
        return len(self.merged_data)
    
    def __getitem__(self, idx):
        """Returns the sample of the dataset at the given index.

        Parameters
        ----------
        index : int
            The index of the sample to return.
            
        Returns
        -------
        features : pandas.Series
            The features of the sample.
        labels : pandas.Series
            The label of the sample. (The wind power production)
        relative_position : pandas.Series
            The relative position of the sample.
        
        Notes
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.merged_data.iloc[idx, self.merged_data.columns != 'Patv'].values  # Exclude labels column
        label = self.merged_data.iloc[idx, -3]
        
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            labels = self.target_transform(labels)
            
        return torch.tensor(sample, dtype=torch.float64), torch.tensor(label, dtype=torch.float64)