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
from copy import deepcopy
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

    def get_keys_based_on_name_data_type_or_input(self, data: dict, column_name: str = None, data_type: str = None, category = None) -> list:
        """
        Returns the keys of the data that are of a specific type, name or category.
        If the column_name is specified, it will return all the keys that contain the column_name in their name. 
        If the data_type is specified, it will return all the keys that contain the data_type in their name.
        If the data_type and the category are specified, it will return all the keys that contain the data_type and the category in their name.
        """

        # Check that one of column_name, data_type or category is not None
        if column_name is None and data_type is None and category is None:
            raise ValueError("At least one of column_name, data_type or category should be specified.")
        
        # Check that category is not the only one specified
        if category is not None and column_name is None and data_type is None:
            raise ValueError("category cannot be the only one specified.")
        
        if column_name is not None:
            return [key for key in data if column_name in key.split(':')[0]]
        if data_type is not None:
            if category is not None:
                return [key for key in data if data_type in key.split(':')[1] and category in key.split(':')[2]]
            else:
                return [key for key in data if data_type in key.split(':')[1]]

 
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

    def noise_dna_uniform_masker(self, data: dict, **noise_params) -> dict:
        """
        Adds noise to the data of a single input.
        Applied on all input keys that have the dna data type.
        """


        dna_type_keys = self.get_keys_based_on_name_data_type_or_input(data, data_type='dna')

        for key in dna_type_keys:
            data[key] = self.dna.add_noise_uniform_text_masker_all_inputs(data[key], seed=self.seed, **noise_params)
        
        return data



