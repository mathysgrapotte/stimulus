"""
This file contains the splitter classes for splitting data accordingly
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

class AbstractSplitter(ABC):
    """
    Abstract class for splitters.
    """

    @abstractmethod
    def get_split_indexes(self, length_of_data: int, split: list, seed: float = None) -> list:
        """
        Splits the data. Always return indices mapping to the original list. 
        """
        raise NotImplementedError
    
    # @abstractmethod
    # def distance(self, data_one: Any, data_two: Any) -> float:
    #     """
    #     Calculates the distance between two elements of the data.
    #     """
    #     raise NotImplementedError
    

class RandomSplitter(AbstractSplitter):
    """
    This splitter randomly splits the data.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_split_indexes(self, data: pl.DataFrame, split: list = [0.7, 0.2, 0.1], seed: float = None) -> list | list | list:
        """
        Splits the data indices into train, validation, and test sets. 
        One can use these lists of indices to parse the data afterwards.

        args:
            data: polars dataframe
                The data loaded with polars.
            split: list
                The proportions for [train, validation, test] splits.
            seed: float
                The seed for reproducibility.
        returns:
            train: list
                The indices for the training set.
            validation: list
                The indices for the validation set.
            test: list
                The indices for the test set.
        """
        if len(split) != 3:
            raise ValueError("The split argument should be a list with length 3 that contains the proportions for [train, validation, test] splits.")
        # Use round to avoid errors due to floating point imprecisions
        if round(sum(split),3) <  1.0:
            raise ValueError("The sum of the split proportions should be 1. Instead, it is {}.".format(sum(split)))

        # compute the length of the data
        length_of_data = len(data)

        # Generate a list of indices and shuffle it
        indices = np.arange(length_of_data)
        np.random.seed(seed)
        np.random.shuffle(indices)

        # Calculate the sizes of the train, validation, and test sets
        train_size = int(split[0] * length_of_data)
        validation_size = int(split[1] * length_of_data)

        # Split the shuffled indices according to the calculated sizes
        train = indices[:train_size].tolist()
        validation = indices[train_size:train_size+validation_size].tolist()
        test = indices[train_size+validation_size:].tolist()

        return train, validation, test
