"""
this file contains classes for various input data types
"""
import encoding.encoders as encoders
import noise.noise_generators as noise_generators
import numpy as np

from abc import ABCmeta, abstractmethod
from typing import Any, Literal

class AbstractType(ABCmeta):
    """
    Abstract class for input types.
    """
    
    @abstractmethod
    def encode(self, data: Any) -> Any:
        """
        By default, returns the data as is. 
        """
        return data
    
    @abstractmethod
    def encode_all(self, data: list) -> list:
        """
        By default, returns the data as is. 
        """
        return [self.encode(d) for d in data]

class Dna(AbstractType):
    """
    class for dealing with DNA data    
    """

    def __init__(self, **parameters) -> None:
        self.one_hot_encoder = encoders.TextOneHotEncoder(alphabet=parameters.get("one_hot_encoder_alphabet", "ACGT"))
        self.uniform_text_masker = noise_generators.UniformTextMasker(probability=parameters.get("text_masker_probability", 0.1), seed=parameters.get("random_seed", 0))
        
    def one_hot_encode(self, data: str) -> np.array:
        """
        Encodes the data of a single input.
        """
        return self.one_hot_encoder.encode(data)

    def one_hot_encode_all(self, data: list) -> list:
        """
        Encodes the data of multiple inputs.
        """
        return self.one_hot_encoder.encode_multiprocess(data)
    
    def encode(self, data: str, encoder: Literal['one_hot'] = 'one_hot') -> Any:
        if encoder == 'one_hot':
            return self.one_hot_encode(data)
        else:
            raise ValueError(f"Unknown encoder {encoder}")


    def encode_all(self, data: list, encoder: Literal['one_hot'] = 'one_hot') -> list:
        if encoder == 'one_hot':
            return self.one_hot_encode_all(data)
        else:
            raise ValueError(f"Unknown encoder {encoder}")

    
    def add_noise_uniform_text_masker(self, data: str) -> str:
        """
        Adds noise to the data of a single input.
        """
        return self.uniform_text_masker.add_noise(data)
    
    def add_noise_uniform_text_masker_all_inputs(self, data: list) -> list:
        """
        Adds noise to the data of multiple inputs.
        """
        return self.uniform_text_masker.add_noise_multiprocess(data)
    

class Float():
    """
    class for dealing with float data
    """
    
    def __init__(self, **parameters) -> None:
        self.gaussian_noise = noise_generators.GaussianNoise(mean=parameters.get("gaussian_noise_mean", 0), std=parameters.get("gaussian_noise_std", 1))

    def add_noise_gaussian_noise(self, data: float) -> float:
        """
        Adds noise to the data of a single input.
        """
        return self.gaussian_noise.add_noise(data)
    
    def add_noise_gaussian_noise_all_inputs(self, data: list) -> list:
        """
        Adds noise to the data of multiple inputs.
        """
        return self.gaussian_noise.add_noise_multiprocess(data)
    
