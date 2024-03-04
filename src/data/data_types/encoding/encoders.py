"""
This file contains encoders classes for encoding various types of data.
"""

from abc import ABC, abstractmethod
from typing import Any
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import multiprocessing as mp

class AbstractEncoder(ABC):
    """
    Abstract class for encoders.
    """

    @abstractmethod
    def encode(self, data: Any) -> Any:
        """
        Encodes the data. 
        This method takes as input a single data point, should be mappable to a single output. 
        """
        raise NotImplementedError
    
    def encode_multiprocess(self, data: list) -> list:
        """
        Encodes the data using multiprocessing.
        """
        with mp.Pool(mp.cpu_count()) as pool:
            return pool.map(self.encode, data)

    @abstractmethod
    def decode(self, data: Any) -> Any:
        """
        Decodes the data.
        """
        raise NotImplementedError
    

class TextOneHotEncoder(AbstractEncoder):
    """
    One hot encoder for text data.
    """

    def __init__(self, alphabet: str = "acgt") -> None:
        self.alphabet = alphabet
        self.encoder = OneHotEncoder(categories=[list(alphabet)], handle_unknown='ignore') # handle_unknown='ignore' unsures that a vector of zeros is returned for unknown characters, such as 'Ns' in DNA sequences

    def _sequence_to_array(self, sequence: str) -> np.array:
        """
        This function transforms the given sequence to an array.
        """
        sequence_lower_case = sequence.lower()
        sequence_array = np.array(list(sequence_lower_case))
        return sequence_array.reshape(-1, 1)

    def encode(self, data: str) -> np.array:
        """
        Encodes the data.
        """
        return self.encoder.fit_transform(self._sequence_to_array(data)).toarray()
    
    def decode(self, data: np.array) -> str:
        """
        Decodes the data.
        """
        return self.encoder.inverse_transform(data)
    
