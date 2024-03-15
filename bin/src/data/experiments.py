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


class AbstractExperiment(ABC):
    """
    Abstract class for experiments.

    WARNING, DATA_TYPES ARGUMENT NAMES SHOULD BE ALL LOWERCASE, CHECK THE DATA_TYPES MODULE FOR THE TYPES THAT HAVE BEEN IMPLEMENTED.
    """
    def __init__(self, seed: float = None) -> None:
        # allow ability to add a seed for reproducibility
        self.seed = seed
        
    def get_function_encode_all(self, data_type: str) -> Any:
        """
        This method gets the encoding function for a specific data type.
        """
        return getattr(self, data_type)['encoder'].encode_all
    
    def get_function_noise_all(self, data_type: str, noise_generator: str) -> Any:
        """
        This method adds noise to all the entries.
        """
        return getattr(self, data_type)['noise_generators'][noise_generator].add_noise_all

    def get_function_split(self, split_method: str) -> Any:
        """
        This method returns the function for splitting the data.
        """
        return self.split[split_method].get_split_indexes
    

class DnaToFloatExperiment(AbstractExperiment):
    """
    Class for dealing with DNA to float predictions (for instance regression from DNA sequence to CAGE value)
    """
    def __init__(self) -> None:
        super().__init__()
        self.dna = {'encoder': encoders.TextOneHotEncoder(alphabet='acgt'), 'noise_generators': {'UniformTextMasker': noise_generators.UniformTextMasker(mask='N')}}
        self.float = {'encoder': encoders.FloatEncoder(), 'noise_generators': {'GaussianNoise': noise_generators.GaussianNoise()}}
        self.split = {'RandomSplitter': spliters.RandomSplitter()}


class ProtDnaToFloatExperiment(DnaToFloatExperiment):
    """
    Class for dealing with Protein and DNA to float predictions (for instance regression from Protein sequence + DNA sequence to binding score)
    """
    def __init__(self) -> None:
        super().__init__()
        self.prot = {'encoder': encoders.TextOneHotEncoder(alphabet='acdefghiklmnpqrstvwy'), 'noise_generators': {'UniformTextMasker': noise_generators.UniformTextMasker(mask='X')}}