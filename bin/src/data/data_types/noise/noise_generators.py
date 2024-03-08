"""
This file contains noise generators classes for generating various types of noise.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import multiprocessing as mp

class AbstractNoiseGenerator(ABC):
    """
    Abstract class for noise generators. 
    All noise function should have the seed in it. Because the multiprocessing of them could unset the seed in short.
    """

    def __init__(self):
        pass

    @abstractmethod
    def add_noise(self, data: Any, seed: float = None) -> Any:
        """
        Adds noise to the data.  
        They should have the following line
        """
        #  np.random.seed(seed)
        raise NotImplementedError
    
    def add_noise_multiprocess(self, data: list, seed: float = None, **noise_params) -> list:
        """
        Adds noise to the data using multiprocessing.
        """
        with mp.Pool(mp.cpu_count()) as pool:
            # reshaping the inputs of this function to meet starmap requirements, basically adding into a tuple the list[elem] + seed
            function_specific_input = [(item, seed) for item in data]
            return pool.starmap(self.add_noise, function_specific_input)
        

class UniformTextMasker(AbstractNoiseGenerator):
    """
    This noise generators replace characters with 'N' with a given probability.
    """


    def add_noise(self, data: str, probability: float = 0.1, seed: float = None) -> str:
        """
        Adds noise to the data.
        """

        np.random.seed(seed)
        return ''.join([c if np.random.rand() > probability else 'N' for c in data])

    def add_noise_multiprocess(self, data: list, probability: float = 0.1, seed: float = None) -> list:
        """
        Adds noise to the data using multiprocessing.
        """

        with mp.Pool(mp.cpu_count()) as pool:
            function_specific_input = [(item, probability, seed) for item in data]
            return pool.starmap(self.add_noise, function_specific_input)

class GaussianNoise(AbstractNoiseGenerator):
    """
    This noise generator adds gaussian noise to float values
    """


    def add_noise(self, data: float, mean: float = 0, std: float= 0, seed: float = None) -> float:
        """
        Adds noise to a single point of data.
        """

        np.random.seed(seed)
        return data + np.random.normal(mean, std)
    
    def add_noise_multiprocess(self, data: list, mean: float = 0, std: float = 0, seed: float = None) -> list:
        """
        Adds noise to the data using np arrays
        # TODO return a np array to gain performance.
        """

        np.random.seed(seed)
        return list(np.array(data) + np.random.normal(mean, std, len(data)))