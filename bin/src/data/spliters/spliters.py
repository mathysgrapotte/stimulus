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
    def __init__(self, seed: float = 0) -> None:
        # allow ability to add a seed for reproducibility
        if seed != 0:
            np.random.seed(seed)

    @abstractmethod
    def split(self, data: list) -> list:
        """
        Splits the data. Always return indices mapping to the original list. 
        """
        raise NotImplementedError
    
    @abstractmethod
    def distance(self, data_one: Any, data_two: Any) -> float:
        """
        Calculates the distance between two elements of the data.
        """
        raise NotImplementedError
    

class RandomSplitter(AbstractSplitter):
    """
    This splitter randomly splits the data.
    """

    def __init__(self, seed: float = 0) -> None:
        super().__init__(seed=seed)

    def split(self, length_of_data: int, split: tuple) -> list | list | list:
        """
        Randomly splits the data in three lists according to the split tuple, the split tuple should contain two values between 0 and 1 in an ascending manner.
        Instead of returning the original data, returns three lists of indexes mapping to the indexes in the original data.
        """
        if split[0] >= split[1]:
            raise ValueError("The split tuple should contain two values between 0 and 1 in an ascending manner.")
        train, test, validation = [], [], []
        for i in range(length_of_data):
            r = np.random.rand()
            if r < split[0]:
                train.append(i)
            elif r < split[1]:
                test.append(i)
            else:
                validation.append(i)
        return train, test, validation