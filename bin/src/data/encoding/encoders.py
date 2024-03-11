"""
This file contains encoders classes for encoding various types of data.
"""

from abc import ABC, abstractmethod
from typing import Any, Union
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
    
    @abstractmethod
    def encode_all(self, data: list) -> Any:
        """
        Encodes the data. 
        This method takes as input a list of data points, should be mappable to a single output. 
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, data: Any) -> Any:
        """
        Decodes the data.
        """
        raise NotImplementedError
    
    def encode_multiprocess(self, data: list) -> list:
        """
        Helper function for encoding the data using multiprocessing.
        """
        with mp.Pool(mp.cpu_count()) as pool:
            return pool.map(self.encode, data)
    

class TextOneHotEncoder(AbstractEncoder):
    """
    One hot encoder for text data.

    NOTE that it will onehot encode based on the alphabet. 
    If there is any character not included in the alphabet, that character will be presented by a vector of zeros.
    """

    def __init__(self, alphabet: str = "acgt") -> None:
        self.alphabet = alphabet
        self.encoder = OneHotEncoder(categories=[list(alphabet)], handle_unknown='ignore') # handle_unknown='ignore' unsures that a vector of zeros is returned for unknown characters, such as 'Ns' in DNA sequences

    def _sequence_to_array(self, sequence: str) -> np.array:
        """
        This function transforms the given sequence to an array.
        eg. 'abcd' -> array(['a'],['b'],['c'],['d'])
        """
        sequence_lower_case = sequence.lower()
        sequence_array = np.array(list(sequence_lower_case))
        return sequence_array.reshape(-1, 1)

    def encode(self, data: str) -> np.array:
        """
        Encodes the data.
        """
        return np.squeeze(np.stack(self.encoder.fit_transform(self._sequence_to_array(data)).toarray()))
    
    def encode_all(self, data: Union[list, str]) -> np.array:
        """
        Encodes the data, if the list is length one, call encode instead.
        It resturns a list with all the encoded data entries.
        """
        # check if the data is a str, in that case it should use the encode sequence method
        if isinstance(data, str):
            return [self.encode(data)]
        else:
            return self.encode_multiprocess(data)
    
    def decode(self, data: np.array) -> str:
        """
        Decodes the data.
        """
        return self.encoder.inverse_transform(data)
    
class FloatEncoder(AbstractEncoder):
    """
    Encoder for float data.
    """
    def encode(self, data: float) -> float:
        """
        Encodes the data. 
        This method takes as input a single data point, should be mappable to a single output. 
        """
        return float(data)
    
    def encode_all(self, data: list) -> list:
        """
        Encodes the data. 
        This method takes as input a list of data points, should be mappable to a single output. 
        """

        # check if data is a string, in that case it should use the encode sequence method
        if isinstance(data, str):
            return [self.encode(data)]
        else:
            return [float(d) for d in data]
    
    def decode(self, data: float) -> float:
        """
        Decodes the data.
        """
        return data
    
    
