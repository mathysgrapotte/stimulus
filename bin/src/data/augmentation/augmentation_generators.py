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
        

class ReverseComplement(AbstractAugmentationGenerator):
    """
    This noise generators replace characters with a masking character with a given probability.
    """
    def __init__(self, type:str = "DNA") -> None:
        if (type != "DNA"):
            raise ValueError("Currently only DNA sequences are supported. Update the class ReverseComplement to support other types.")
        if type == "DNA":
            self.complement_mapping = str.maketrans('ATCG', 'TAGC')


    def add_augmentation(self, data: str) -> str:
        """
        Returns the reverse complement of a list of string data using the complement_mapping.
        """
        return data.translate(self.complement_mapping)[::-1]

    def add_augmentation_all(self, data: list) -> list:
        """
        Adds reverse complement to the data using multiprocessing.
        """
        with mp.Pool(mp.cpu_count()) as pool:
            function_specific_input = [(item) for item in data]
            return pool.starmap(self.add_augmentation, function_specific_input)
        


    
