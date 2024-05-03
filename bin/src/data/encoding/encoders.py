"""
This file contains encoders classes for encoding various types of data.
"""

from abc import ABC, abstractmethod
from typing import Any, Union
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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
    One hot encoder for text data

    NOTE encodes based on the given alphabet
    If a character c is not in the alphabet, c will be represented by a vector of zeros.
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
    
    def encode_all(self, data: Union[list, str]) -> Union[np.array, list]:
        """
        It encodes all the data.
        If only one entry is given, it will call the encode method directly.
        If a list of many entries is given, it will call the encode_multiprocess method.

        TODO instead maybe we can run encode_multiprocess when data size is larger than a certain threshold.
        """
        if not isinstance(data, list):
            encoded_data = self.encode(data)
            return np.array([encoded_data])  # reshape the array in a batch of 1 configuration as a np.ndarray (so shape is (1, sequence_length, alphabet_length))

        else:
            encoded_data = self.encode_multiprocess(data)
            # try to transform the list of arrays to a single array and return it 
            # if it fails (when the list of arrays is not of the same length), return the list of arrays
            try:
                return np.array(encoded_data)
            except ValueError:
                return encoded_data
    
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
    
    def encode_all(self, data: list) -> np.array:
        """
        Encodes the data. 
        This method takes as input a list of data points, should be mappable to a single output. 
        """
        if not isinstance(data, list):
            data = [data]
        return np.array([self.encode(d) for d in data])
    
    def decode(self, data: float) -> float:
        """
        Decodes the data.
        """
        return data
    
class IntEncoder(FloatEncoder):
    """
    Encoder for integer data.
    """
    def encode(self, data: int) -> int:
        """
        Encodes the data. 
        This method takes as input a single data point, should be mappable to a single output. 
        """
        return int(data)
    
class StrClassificationIntEncoder(AbstractEncoder):
    """
    Considering a ensemble of strings, this encoder encodes them into integers from 0 to (n-1) where n is the number of unique strings.
    """

    def encode(self, data: str) -> int:
        """
        Returns an error since encoding a single string does not make sense.
        """

        raise NotImplementedError("Encoding a single string does not make sense. Use encode_all instead.")
    
    def encode_all(self, data: list) -> np.array:
        """
        Encodes the data. 
        This method takes as input a list of data points, should be mappable to a single output, using LabelEncoder from scikit learn and returning a numpy array.
        For more info visit : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html 
        """

        if not isinstance(data, list):
            data = [data]
        encoder = LabelEncoder()
        return encoder.fit_transform(data)
    
    def decode(self, data: Any) -> Any:
        """
        Returns an error since decoding does not make sense without encoder information, which is not yet supported.
        """

        raise NotImplementedError("Decoding is not yet supported for StrClassificationInt.")
    
class StrClassificationScaledEncoder(StrClassificationIntEncoder):
    """
    Considering a ensemble of strings, this encoder encodes them into floats from 0 to 1 (essentially scaling the integer encoding).
    """

    def encode_all(self, data: list) -> np.array:
        """
        Encodes the data. 
        This method takes as input a list of data points, should be mappable to a single output, using LabelEncoder from scikit learn and returning a numpy array.
        For more info visit : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html 
        """

        encoded_data = super().encode_all(data)
        return encoded_data / (len(np.unique(encoded_data)) - 1)
    
    def decode(self, data: Any) -> Any:
        """
        Returns an error since decoding does not make sense without encoder information, which is not yet supported.
        """

        raise NotImplementedError("Decoding is not yet supported for StrClassificationScaled.")
    
class FloatRankEncoder(AbstractEncoder):
    """
    Considering an ensemble of float values, this encoder encodes them into floats from 0 to 1, where 1 is the maximum value and 0 is the minimum value.
    """

    def encode(self, data: float) -> float:
        """
        Returns an error since encoding a single float does not make sense.
        """

        raise NotImplementedError("Encoding a single float does not make sense. Use encode_all instead.")
    
    def encode_all(self, data: list) -> np.array:
        """
        Encodes the data. 
        This method takes as input a list of data points, should be mappable to a single output, using min-max scaling.
        """

        if not isinstance(data, list):
            data = [data]
        data = np.array(data)
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    def decode(self, data: Any) -> Any:
        """
        Returns an error since decoding does not make sense without encoder information, which is not yet supported.
        """

        raise NotImplementedError("Decoding is not yet supported for FloatRank.")
    
class IntRankEncoder(FloatRankEncoder):
    """
    Considering an ensemble of integer values, this encoder encodes them into floats from 0 to 1, where 1 is the maximum value and 0 is the minimum value.
    """

    def encode(self, data: int) -> int:
        """
        Returns an error since encoding a single integer does not make sense.
        """

        raise NotImplementedError("Encoding a single integer does not make sense. Use encode_all instead.")
    
    def encode_all(self, data: list) -> np.array:
        """
        Encodes the data. 
        This method takes as input a list of data points, should be mappable to a single output, using min-max scaling.
        """

        if not isinstance(data, list):
            data = [data]
        data = np.array(data)
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    def decode(self, data: Any) -> Any:
        """
        Returns an error since decoding does not make sense without encoder information, which is not yet supported.
        """

        raise NotImplementedError("Decoding is not yet supported for IntRank.")

