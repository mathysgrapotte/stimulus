"""
This file contains noise generators classes for generating various types of noise.
"""

from abc import ABCmeta, abstractmethod
from typing import Any
import numpy as np
import multiprocessing as mp

class AbstractNoiseGenerator(ABCmeta):
    """
    Abstract class for noise generators.
    """

    def __init__(self, seed: float = 0) -> None:
        # allow ability to add a seed for reproducibility
        if seed != 0:
            np.random.seed(seed)
        

    @abstractmethod
    def add_noise(self, data: Any) -> Any:
        """
        Adds noise to the data.
        """
        raise NotImplementedError
    
    def add_noise_multiprocess(self, data: list) -> list:
        """
        Adds noise to the data using multiprocessing.
        """
        with mp.Pool(mp.cpu_count()) as pool:
            return pool.map(self.add_noise, data)
        

class UniformTextMasker(AbstractNoiseGenerator):
    """
    This noise generators replace characters with 'N' with a given probability.
    """

    def __init__(self, probability: float = 0.1, seed: float = 0) -> None:
        super().__init__(seed=seed)
        self.probability = probability


    def add_noise(self, data: str) -> str:
        """
        Adds noise to the data.
        """
        return ''.join([c if np.random.rand() > self.probability else 'N' for c in data])
    
class GaussianNoise(AbstractNoiseGenerator):
    """
    This noise generator adds gaussian noise to float values
    """

    def __init__(self, mean: float = 0, std: float = 1, seed: float = 0) -> None:
        super().__init__(seed=seed)
        self.mean = mean
        self.std = std

    def add_noise(self, data: float) -> float:
        """
        Adds noise to a single point of data.
        """

        return data + np.random.normal(self.mean, self.std)
    
    def add_noise_multiprocess(self, data: list) -> list:
        """
        Adds noise to the data using np arrays
        """
        return list(np.array(data) + np.random.normal(self.mean, self.std, len(data)))
