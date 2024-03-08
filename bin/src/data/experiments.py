"""
Experiments are classes parsed by CSV master classes to run experiments. 
Conceptually, experiment classes contain data types, transformations etc and are used to duplicate the input data into many datasets. 
Here we provide standard experiments as well as an absctract class for users to implement their own. 


# TODO implement noise schemes and splitting schemes.
"""

from abc import ABC, abstractmethod
from typing import Any
from .data_types import data_types as data_types
from .spliters import spliters as spliters
import numpy as np

class AbstractExperiment(ABC):
    """
    Abstract class for experiments.

    WARNING, DATA_TYPES ARGUMENT NAMES SHOULD BE ALL LOWERCASE, CHECK THE DATA_TYPES MODULE FOR THE TYPES THAT HAVE BEEN IMPLEMENTED.
    """
    def __init__(self, seed: float = None) -> None:
        # allow ability to add a seed for reproducibility
        self.seed = seed

        #self.random_splitter = spliters.RandomSplitter(seed=seed) 


    def get_split_indexes(self, data: list, split: tuple) -> list | list | list:
        """
        Returns the indexes of the split data.
        """
        raise NotImplementedError

 
    def noise(self, data: Any, noise_method: str, **noise_params: dict) -> Any:
        """
        Adds noise to the data, using function defined in self.noise
        """
        # check if noise_method exist in the class, if it does, call it with the associated **noise_params, if not raise an error

        if hasattr(self, noise_method):
            return getattr(self, noise_method)(data, **noise_params)
        else:
            raise NotImplementedError(f"No noise method {noise_method} in the class {self.__class__.__name__}")

    
class DnaToFloatExperiment(AbstractExperiment):
    """
    Class for dealing with DNA to float predictions (for instance regression from DNA sequence to CAGE value)
    """

    def __init__(self, seed: float = None, **parameters) -> None:
        super().__init__(seed)
        self.dna = data_types.Dna(**parameters)
        self.float = data_types.Float(**parameters)

    def add_noise(self, data: list) -> list:
        """
        Adds noise to the data of a single input.
        """
        return self.dna.add_noise_uniform_text_masker_all_inputs(data, seed=self.seed)
    
    def noise_scheme(self, data: list, params: dict) -> dict:
        output = {}
        for key in params: 
            output[key] = self.add_noise(data, params[key])

        return output



