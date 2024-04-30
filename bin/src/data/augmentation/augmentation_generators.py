"""
This file contains augmentation generators classes for generating various types of data augmentation.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import multiprocessing as mp

class AbstractAugmentationGenerator(ABC):
    """
    Abstract class for augmentation generators. 
    All augmentation function should have the seed in it. Because the multiprocessing of them could unset the seed in short.
    """

    def __init__(self):
        pass

    @abstractmethod
    def add_augmentation(self, data: Any, seed: float = None) -> Any:
        """
        Adds noise to the data.  
        They should have the following line
        """
        #  np.random.seed(seed)
        raise NotImplementedError
    
    @abstractmethod
    def add_augmentation_all(self, data: list, seed: float = None) -> list:
        """
        Adds noise to the data.
        """
        #  np.random.seed(seed)
        raise NotImplementedError
        


    
