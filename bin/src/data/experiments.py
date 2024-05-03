"""
Experiments are classes parsed by CSV master classes to run experiments. 
Conceptually, experiment classes contain data types, transformations etc and are used to duplicate the input data into many datasets. 
Here we provide standard experiments as well as an absctract class for users to implement their own. 


# TODO implement noise schemes and splitting schemes.
"""

from abc import ABC, abstractmethod
from typing import Any
from .splitters import splitters as splitters
from .encoding import encoders as encoders
from .transform import data_transformation_generators as data_transformation_generators


class AbstractExperiment(ABC):
    """
    Abstract class for experiments.

    WARNING, DATA_TYPES ARGUMENT NAMES SHOULD BE ALL LOWERCASE, CHECK THE DATA_TYPES MODULE FOR THE TYPES THAT HAVE BEEN IMPLEMENTED.
    """
    def __init__(self, seed: float = None) -> None:
        # allow ability to add a seed for reproducibility
        self.seed = seed
        # added because if the user does not define this it does not crach the get_function_split, random split works for every class afteralll
        self.split = {'RandomSplitter': splitters.RandomSplitter()}
        
    def get_function_encode_all(self, data_type: str) -> Any:
        """
        This method gets the encoding function for a specific data type.
        """
        return getattr(self, data_type)['encoder'].encode_all

    def get_data_transformer(self, data_type: str, transformation_generator: str) -> Any:
        """
        This method transforms the data (noising, data augmentation etc).
        """
        return getattr(self, data_type)['data_transformation_generators'][transformation_generator]

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
        self.dna = {'encoder': encoders.TextOneHotEncoder(alphabet='acgt'), 'data_transformation_generators': {'UniformTextMasker': data_transformation_generators.UniformTextMasker(mask='N'), 'ReverseComplement': data_transformation_generators.ReverseComplement()}}
        self.float = {'encoder': encoders.FloatEncoder(), 'data_transformation_generators': {'GaussianNoise': data_transformation_generators.GaussianNoise()}}
        self.split = {'RandomSplitter': splitters.RandomSplitter()}        


class ProtDnaToFloatExperiment(DnaToFloatExperiment):
    """
    Class for dealing with Protein and DNA to float predictions (for instance regression from Protein sequence + DNA sequence to binding score)
    """
    def __init__(self) -> None:
        super().__init__()
<<<<<<< HEAD
        self.prot = {'encoder': encoders.TextOneHotEncoder(alphabet='acdefghiklmnpqrstvwy'), 'noise_generators': {'UniformTextMasker': noise_generators.UniformTextMasker(mask='X')}}

class TitanicExperiment(AbstractExperiment):
    """
    Class for dealing with the Titanic dataset as a test format.
    """

    def __init__(self) -> None:
        super().__init__()
        self.int_class = {'encoder': encoders.IntEncoder(), 'noise_generators': {}}
        self.str_class = {'encoder': encoders.StrClassificationIntEncoder(), 'noise_generators': {}}
        self.int_reg = {'encoder': encoders.IntRankEncoder(), 'noise_generators': {}}
        self.float_rank = {'encoder': encoders.FloatRankEncoder(), 'noise_generators': {}}
=======
        self.prot = {'encoder': encoders.TextOneHotEncoder(alphabet='acdefghiklmnpqrstvwy'), 'data_transformation_generators': {'UniformTextMasker': data_transformation_generators.UniformTextMasker(mask='X')}}
>>>>>>> main
