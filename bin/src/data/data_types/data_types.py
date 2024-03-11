"""
this file contains classes for various input data types
"""
from .encoding import encoders as encoders
from .noise import noise_generators as noise_generators
import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Literal

class AbstractType(ABC):
    """
    Abstract class for input types.
    """
    
    @abstractmethod
    def encode(self, data: Any) -> Any:
        """
        If no encoding, return the data as is. 
        """
        return NotImplementedError
    
    @abstractmethod
    def encode_all(self, data: list) -> np.array:
        """
        If no encoding, return the data as is.
        """
        return NotImplementedError

class Dna(AbstractType):
    """
    class for dealing with DNA data
    # TODO make a Text base class for this class and other text based classes (rna, protein etc...)    
    """

    def __init__(self, **parameters) -> None:
        self.one_hot_encoder = encoders.TextOneHotEncoder(alphabet=parameters.get("one_hot_encoder_alphabet", "acgt"))
        self.uniform_text_masker = noise_generators.UniformTextMasker()
        
    def one_hot_encode(self, data: str) -> np.array:
        """
        Encodes the data of a single input.
        """
        return self.one_hot_encoder.encode(data)

    def one_hot_encode_all(self, data: list) -> list:
        """
        Encodes the data of multiple inputs.
        """
        return self.one_hot_encoder.encode_all(data)
    
    def encode(self, data: str, encoder: Literal['one_hot'] = 'one_hot') -> Any: #TODO call from get attribute instead of using if else
        if encoder == 'one_hot':
            return self.one_hot_encode(data)
        else:
            raise ValueError(f"Unknown encoder {encoder}")

    def encode_all(self, data: list, encoder: Literal['one_hot'] = 'one_hot') -> list[np.array]:
        if encoder == 'one_hot':
            return self.one_hot_encode_all(data)
        else:
            raise ValueError(f"Unknown encoder {encoder}")

    def add_noise_uniform_text_masker(self, data: str, seed: float = None, **noise_params) -> str:
        """
        Adds noise to the data of a single input.
        """
        # get the probability param from noise_params, default value is set to 0.1
        probability = noise_params.get("probability", 0.1)
        return self.uniform_text_masker.add_noise(data, probability=probability, mask='N', seed=seed)
    
    def add_noise_uniform_text_masker_all_inputs(self, data: list, seed: float = None, **noise_params) -> list:
        """
        Adds noise to the data of multiple inputs.
        """
        # get the probability param from noise_params, default value is set to 0.1 
        probability = noise_params.get("probability", 0.1)
        return self.uniform_text_masker.add_noise_multiprocess(data, probability=probability, mask='N', seed=seed)
    

class Prot(AbstractType):
    """
    class for dealing with protein data
    """

    def __init__(self, **parameters) -> None:
        self.one_hot_encoder = encoders.TextOneHotEncoder(alphabet=parameters.get("one_hot_encoder_alphabet", "acdefghiklmnpqrstvwy"))
        self.uniform_text_masker = noise_generators.UniformTextMasker()
        
    def one_hot_encode(self, data: str) -> np.array:
        """
        Encodes the data of a single input.
        """
        return self.one_hot_encoder.encode(data)

    def one_hot_encode_all(self, data: list) -> list:
        """
        Encodes the data of multiple inputs.
        """
        return self.one_hot_encoder.encode_all(data)
    
    def encode(self, data: str, encoder: Literal['one_hot'] = 'one_hot') -> Any: #TODO call from get attribute instead of using if else
        if encoder == 'one_hot':
            return self.one_hot_encode(data)
        else:
            raise ValueError(f"Unknown encoder {encoder}")

    def encode_all(self, data: list, encoder: Literal['one_hot'] = 'one_hot') -> list[np.array]:
        if encoder == 'one_hot':
            return self.one_hot_encode_all(data)
        else:
            raise ValueError(f"Unknown encoder {encoder}")

    def add_noise_uniform_text_masker(self, data: str, seed: float = None, **noise_params) -> str:
        """
        Adds noise to the data of a single input.
        """
        # get the probability param from noise_params, default value is set to 0.1
        probability = noise_params.get("probability", 0.1)
        return self.uniform_text_masker.add_noise(data, probability=probability, mask='X', seed=seed)
    
    def add_noise_uniform_text_masker_all_inputs(self, data: list, seed: float = None, **noise_params) -> list:
        """
        Adds noise to the data of multiple inputs.
        """
        # get the probability param from noise_params, default value is set to 0.1 
        probability = noise_params.get("probability", 0.1)
        return self.uniform_text_masker.add_noise_multiprocess(data, probability=probability, mask='X', seed=seed)
    

class Float():
    """
    class for dealing with float data
    """
    def __init__(self) -> None:
        self.gaussian_noise = noise_generators.GaussianNoise()

    def add_noise_gaussian_noise(self, data: float, seed: float = None) -> float:
        """
        Adds noise to the data of a single input.
        """
        return self.gaussian_noise.add_noise(data, seed=seed)
    
    def add_noise_gaussian_noise_all_inputs(self, data: list, seed: float = None) -> list:
        """
        Adds noise to the data of multiple inputs.
        """
        return self.gaussian_noise.add_noise_multiprocess(data, seed=seed)
    
    def encode(self, data: Any) -> float:
        return float(data)
    
    def encode_all(self, data: list) -> list[np.array]:
        return [np.array(float(d)) for d in data]
    