"""
Experiments are classes parsed by CSV master classes to run experiments. 
Conceptually, experiment classes contain data types, transformations etc and are used to duplicate the input data into many datasets. 
Here we provide standard experiments as well as an absctract class for users to implement their own. 


# TODO for next time, I need to add a way of encoding data based on the data type and encoder specified by the experiment class. Then the getitem has to return a dictionary transformed by the encoding functions of the correct class based on the dictionary keys.
-> we can use getstate
"""

from abc import ABC, abstractmethod
from typing import Any
import data_types.data_types as data_types
import spliters.spliters as spliters
import numpy as np

class AbstractExperiment(ABC):
    """
    Abstract class for experiments.
    """
    def __init__(self, data: dict, seed: float = 0) -> None:
        # allow ability to add a seed for reproducibility
        if seed != 0:
            np.random.seed(seed)

        self.random_splitter = spliters.RandomSplitter(seed=seed) 
        self.input, self.label, self.meta = self.split_input_label_meta(data)

    
    def split_input_label_meta(self, data: dict) -> dict | dict | dict:
        """
        The dict data has keys of this form : name:category:type . The category corresponds to input, label or meta. 
        This function splits the data into three dictionaries accordingly, one for each category. 
        The keys of each new dictionary are in this form name:type.
        """
        input_data, label_data, meta_data = {}, {}, {}
        for key in data:
            name, category, data_type = key.split(":")
            if category == "input":
                input_data[f"{name}:{data_type}"] = data[key]
            elif category == "label":
                label_data[f"{name}:{data_type}"] = data[key]
            elif category == "meta":
                meta_data[f"{name}:{data_type}"] = data[key]
            else:
                raise ValueError(f"Unknown category {category}")
        return input_data, label_data, meta_data


    def link_processing(self, data: dict) -> Any:
        """
        Links the processing scheme to each data type.
        Returns a dictionnary of encoding function, otherwise take the identity function (returns the same element) for each type name.
        """
        return {key: getattr(self, key) for key in data}
    
    def get(self, idx: int) -> dict | dict:
        """
        Returns the data at a given index, in the form x, y where x is the input and y the output. 
        Th
        """
        return {key: self.input[key][idx] for key in self.input}, {key: self.label[key][idx] for key in self.label}


    def get_split_indexes(self, data: list, split: tuple) -> list | list | list:
        """
        Returns the indexes of the split data.
        """
        raise NotImplementedError

 
    def noise(self, data: Any) -> Any:
        """
        Adds noise to the data.
        """
        raise NotImplementedError


class DnaToFloatExperiment(AbstractExperiment):
    """
    Class for dealing with DNA to float predictions (for instance regression from DNA sequence to CAGE value)
    """

    def __init__(self, **parameters) -> None:
        self.dna = data_types.Dna(**parameters)
        self.float = data_types.Float(**parameters)

    def one_hot_encode(self, data: list) -> list:
        """
        Encodes the data of a single input.
        """
        return self.dna.one_hot_encode_all_inputs(data)

    def add_noise(self, data: list) -> list:
        """
        Adds noise to the data of a single input.
        """
        return self.dna.add_noise_uniform_text_masker_all_inputs(data)

    def split(self, data: list, split: tuple = (0.7, 0.85)) -> list | list | list:
        """
        Splits the data into train test and validation sets based on the DNA.
        """
        return self.dna.split_random_splitter(data, split)
    


