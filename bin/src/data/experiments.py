"""
Experiments are classes parsed by CSV master classes to run experiments. 
Conceptually, experiment classes contain data types, transformations etc and are used to duplicate the input data into many datasets. 
Here we provide standard experiments as well as an absctract class for users to implement their own. 


# TODO implement noise schemes and splitting schemes.
"""

from abc import ABC, abstractmethod
from typing import Any
from .spliters import spliters as spliters
from .encoding import encoders as encoders
from .noise import noise_generators as noise_generators
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


    def get_split_indexes(self, data: list, split: tuple) -> list | list | list:
        """
        Returns the indexes of the split data.
        """
        raise NotImplementedError
        
    def get_encoding_all(self, data_type: str) -> Any:
        """
        This method gets the encoding function for a specific data type.
        """
        return getattr(self, data_type)['encoder'].encode_all
    
    def add_noise_all(self, data_type, noise_generator: str) -> list:
        """
        This method adds noise to all the entries.
        """
        raise getattr(self, data_type)['noise_generators'][noise_generator].add_noise_all
    
class DnaToFloatExperiment(AbstractExperiment):
    """
    Class for dealing with DNA to float predictions (for instance regression from DNA sequence to CAGE value)
    """
    def __init__(self):
        super().__init__()
        self.dna = {'encoder': encoders.TextOneHotEncoder(alphabet='acgt'), 'noise_generators': {'uniform_text_masker': noise_generators.UniformTextMasker()}}
        self.float = {'encoder': encoders.FloatEncoder(), 'noise_generators': {'uniform_float_masker': noise_generators.GaussianNoise()}}
        #self.protein = {'encoder': encoders.TextOneHotEncoder(), 'noise_generators': {'uniform_text_masker': noise_generators.UniformTextMasker()}}

