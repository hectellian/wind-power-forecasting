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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CustomWindFarmDataset(torch.utils.data.Dataset):
    """A class to load the wind power production data.

    Attributes
    ----------
    data : pandas.DataFrame
        The data loaded from the data file.
    relative_positions : pandas.DataFrame
        The relative positions loaded from the relative position file.
    q : int
        The length of the sequence.
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
    def __init__(self, data_dir, relative_position_file, q=1, device='cpu', transform=None, target_transform=None):
        """Constructs all the necessary attributes for the Dataset object.

        Parameters
        ----------
        data_dir : str
            The path to the data file.
        relative_position_file : str
            The path to the relative position file.
        q : int, optional
            The length of the sequence.
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
        self.q = q
        self.device = device
        self.transform = transform
        self.target_transform = target_transform

        # Merge the data and relative positions
        #self.merged_data = pd.merge(self.data, self.relative_positions, on='TurbID')

        # Convert days to continuous minutes
        self.data["Tmstamp"] = self.data["Day"].astype(str) + " " + self.data["Tmstamp"].astype(str)
        self.data["Tmstamp"] = pd.to_datetime(self.data["Tmstamp"], format='%j %H:%M')
        self.data["Tmstamp"] = self.data["Tmstamp"].rank(method='dense').astype(int) 
        
        self.data.drop(columns=["Day"], inplace=True)
        
        print("Cleaning data...")
        
        # Handle missing values
        print("Removing rows with missing values...")
        self.data = self.data.dropna() # Remove rows with missing values
        print("Rows with missing values removed.")
        
        # Handle Unkown values
        print("Removing rows with unknown values...")
        self.data.drop(self.data[(self.data["Patv"] <= 0) & (self.data["Wspd"] > 2.5)].index, inplace=True)
        self.data.drop(self.data[(self.data["Pab1"] > 89) | (self.data["Pab2"] > 89) | (self.data["Pab3"] > 89)].index, inplace=True)
        print("Rows with unknown values removed.")

        # Handle Abnormal values
        print("Removing rows with abnormal values...")
        self.data.drop(self.data[(self.data["Ndir"] < -720) & (self.data["Ndir"] > 720)].index, inplace=True)
        self.data.drop(self.data[(self.data["Wdir"] < -180) & (self.data["Wdir"] > 180)].index, inplace=True)
        print("Rows with abnormal values removed.")

        self.data.iloc[:, -2:] = self.data.iloc[:, -2:].clip(lower=0) # Replace negative values with 0
        
        self.data.reset_index(drop=True, inplace=True)
        
        if transform is not None: 
            self.data.iloc[:, :-1] = self.transform(self.data.iloc[:, :-1].values)
        if target_transform is not None:
            self.data.iloc[:, -1] = self.target_transform(self.data.iloc[:, -1].values.reshape(-1, 1))
        
        print("Cleaning done.")
        
    def features_size(self):
        """Return the number of columns in the dataset.

        Returns
        -------
        int
            The number of columns in the dataset.
        """
        return len(self.data.columns) - 1

    def correlations(self, target):
        """Return the correlations of all the sequence with the target feature
        
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
    
    def clean_correlations(self, target, threshold):
        """Remove the features with a correlation lower than the threshold
        
        Parameters
        ----------
        target : srt
            target column name
        threshold : float
            The threshold to use to remove the outliers.
        """
        corr_matrix = self.correlations(target)
        corr_matrix = corr_matrix[np.abs(corr_matrix) > threshold]
        self.data = self.data[corr_matrix.index]
        
    
    def plot_correlations(self, target, threshold=None):
        """Plot the correlations of all the sequence with the target feature
        
        Parameters
        ----------
        target : srt
            target column name
        threshold : float, optional
            The threshold to use to remove the outliers.
        """
        target_correlations = self.correlations(target)
            
        target_correlations.plot.barh(color=['#81b5a8' if abs(x) > threshold else '#3f4853' for x in target_correlations])
        plt.title("Correlations with " + target + "\n" + "Threshold: " + str(threshold))
        plt.axvline(x=0, color='k', linewidth=1)
        plt.axvline(x=threshold, color='r', linestyle='--', linewidth=1)
        plt.axvline(x=-threshold, color='r', linestyle='--', linewidth=1)
        plt.xlabel('Features', fontsize=16)
        plt.xlabel(f'Correlation with {target}', fontsize=16)
        plt.show()
    
    def __len__(self):
        """Returns the number of samples.
        """
        return len(self.data) // self.q
    
    def __getitem__(self, idx):
        """Returns the sample of the dataset at the given index.

        Parameters
        ----------
        index : int
            The index of the sample to return.
            
        Returns
        -------
        sequence : pandas.Series
            The sequence features of the sample.
        target : pandas.Series
            The label of the sample. (The wind power production)
        relative_position : pandas.Series
            The relative position of the sample.
        
        Notes
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequence = self.data.loc[idx:idx + self.q - 1, ~self.data.columns.isin(['Patv'])].values # Exclude target column
        target = self.data.loc[idx:idx + self.q - 1, 'Patv'].values
            
        return torch.tensor(sequence, dtype=torch.float, device=self.device), torch.tensor(target, dtype=torch.float, device=self.device)