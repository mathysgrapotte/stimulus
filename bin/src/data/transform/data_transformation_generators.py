"""
This file contains noise generators classes for generating various types of noise.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import multiprocessing as mp


class AbstractDataTransformer(ABC):
    """
    Abstract class for data transformers. 
    All data transformers should have the seed in it. Because the multiprocessing of them could unset the seed in short.
    """

    def __init__(self):
        self.modify_column = None
        self.add_row = None
        self.training_data_only = False

    @abstractmethod
    def transform(self, data: Any, seed: float = None) -> Any:
        """
        Transforms the data.
        """
        #  np.random.seed(seed)
        raise NotImplementedError
    
    @abstractmethod
    def transform_all(self, data: list, seed: float = None) -> list:
        """
        Transforms the data.
        """
        #  np.random.seed(seed)
        raise NotImplementedError


class AbstractNoiseGenerator(AbstractDataTransformer):
    """
    Abstract class for noise generators. 
    All noise function should have the seed in it. Because the multiprocessing of them could unset the seed in short.
    """

    def __init__(self):
        super().__init__()
        self.modify_column = True
        self.add_row = False
        self.training_data_only = True
 

class AbstractAugmentationGenerator(AbstractDataTransformer):
    """
    Abstract class for augmentation generators. 
    All augmentation function should have the seed in it. Because the multiprocessing of them could unset the seed in short.
    """

    def __init__(self):
        super().__init__()
        self.modify_column = False
        self.add_row = True
        self.training_data_only = False


class UniformTextMasker(AbstractNoiseGenerator):
    """
    This noise generators replace characters with a masking character with a given probability.
    """
    def __init__(self, mask: str) -> None:
        super().__init__()
        self.mask = mask

    def transform(self, data: str, probability: float = 0.1, seed: float = None) -> str:
        """
        Adds noise to the data.
        """
        np.random.seed(seed)
        return ''.join([c if np.random.rand() > probability else self.mask for c in data])

    def transform_all(self, data: list, probability: float = 0.1, seed: float = None) -> list:
        """
        Adds noise to the data using multiprocessing.
        """
        with mp.Pool(mp.cpu_count()) as pool:
            function_specific_input = [(item, probability, seed) for item in data]
            return pool.starmap(self.transform, function_specific_input)
        

class GaussianNoise(AbstractNoiseGenerator):
    """
    This noise generator adds gaussian noise to float values
    """

    def transform(self, data: float, mean: float = 0, std: float= 0, seed: float = None) -> float:
        """
        Adds noise to a single point of data.
        """
        np.random.seed(seed)
        return data + np.random.normal(mean, std)
    
    def transform_all(self, data: list, mean: float = 0, std: float = 0, seed: float = None) -> list:
        """
        Adds noise to the data using np arrays
        # TODO return a np array to gain performance.
        """
        np.random.seed(seed)
        return list(np.array(data) + np.random.normal(mean, std, len(data)))
    


class ReverseComplement(AbstractAugmentationGenerator):
    """
    This noise generators replace characters with a masking character with a given probability.
    """
    def __init__(self, type:str = "DNA") -> None:
        super().__init__()
        if (type != "DNA"):
            raise ValueError("Currently only DNA sequences are supported. Update the class ReverseComplement to support other types.")
        if type == "DNA":
            self.complement_mapping = str.maketrans('ATCG', 'TAGC')


    def transform(self, data: str) -> str:
        """
        Returns the reverse complement of a list of string data using the complement_mapping.
        """
        return data.translate(self.complement_mapping)[::-1]

    def transform_all(self, data: list) -> list:
        """
        Adds reverse complement to the data using multiprocessing.
        """
        with mp.Pool(mp.cpu_count()) as pool:
            function_specific_input = [(item) for item in data]
            return pool.map(self.transform, function_specific_input)
        


    
