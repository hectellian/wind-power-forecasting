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
    device : str, optional
        The device to use for the data.
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
    def __init__(self, data_dir, relative_position_file, device='cpu', transform=None, target_transform=None):
        """Constructs all the necessary attributes for the Dataset object.

        Parameters
        ----------
        data_dir : str
            The path to the data file.
        relative_position_file : str
            The path to the relative position file.
        device : str, optional
            The device to use for the data.
        transform : callable, optional
            A function/transform that takes in a sample and returns a transformed
            version.
        target_transform : callable, optional
            A function/transform that takes in the target and transforms it.
        """
        self.data = pd.read_csv(data_dir)
        self.relative_positions = pd.read_csv(relative_position_file)
        self.device = device
        self.transform = transform
        self.target_transform = target_transform

        # Merge the data and relative positions
        #self.merged_data = pd.merge(self.data, self.relative_positions, on='TurbID')

        # Convert days to continuous minutes
        self.data["Tmstamp"] = pd.to_datetime(self.data["Tmstamp"], format='%H:%M')
        self.data["Tmstamp"] = self.data["Tmstamp"].dt.hour * 60 + self.data["Tmstamp"].dt.minute
        self.data = self.data.drop('Day', axis=1)

        # Handle missing values
        self.data = self.data.dropna() # Remove rows with missing values

        # Handle Unkown values
        self.data.drop(self.data[(self.data["Patv"] <= 0) & (self.data["Wspd"] > 2.5)].index)
        self.data.drop(self.data[(self.data["Pab1"] > 89) | (self.data["Pab2"] > 89) | (self.data["Pab3"] > 89)].index)

        # Handle Abnormal values
        self.data.drop(self.data[(self.data["Ndir"] < -720) & (self.data["Ndir"] > 720)].index)
        self.data.drop(self.data[(self.data["Wdir"] < -180) & (self.data["Wdir"] > 180)].index)

        self.data.iloc[:, -2:] = self.data.iloc[:, -2:].clip(lower=0) # Replace negative values with 0

    def correlations(self, target):
        """Return the correlations of all the features with the target feature
        
        Parameters
        ----------
        target : srt
            target column name
            
        Returns
        -------
        target_correlations : pandas.Series
            correlation with target feature
        """
        correlation_matrix = self.data.corr()
        target_correlations = correlation_matrix[target]

        return target_correlations
    
    def __len__(self):
        """Returns the number of samples.
        """
        return len(self.data)
    
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

        sequence = self.data.iloc[idx, :-1].values  # Exclude labels column
        target = self.data.iloc[idx, -1]
        
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            labels = self.target_transform(labels)
            
        return torch.tensor(sequence, dtype=torch.float, device=self.device), torch.tensor(target, dtype=torch.float, device=self.device)