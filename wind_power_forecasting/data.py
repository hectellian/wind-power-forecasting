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

class WindDataset(torch.utils.data.Dataset):
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
    
    def __len__(self):
        """Returns the number of samples.
        """
        return len(self.data)
    
    def __getitem__(self, index):
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
        labels = self.data.iloc[index, -1]
        features = self.data.iloc[index, :-1]
        relative_position = self.relative_positions.iloc[index, :]
        
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            labels = self.target_transform(labels)
            
        return features, labels, relative_position