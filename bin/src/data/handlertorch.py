"""
This file provides the class API for handling the data in pytorch using the Dataset and Dataloader classes
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .csv import CsvLoader
from typing import Any, Tuple, Union, Literal

class TorchDataset(Dataset):
    """
    Class for creating a torch dataset
    """
    def __init__(self, csvpath: str, experiment: Any, split: Tuple[None, int] = None) -> None:
        self.input, self.label, self.meta, self.length = CsvLoader(experiment, csvpath, split=split).get_all_items_and_length()
        self.input, self.label = self.convert_dict_to_dict_of_tensors(self.input), self.convert_dict_to_dict_of_tensors(self.label)

    def convert_to_tensor(self, data: Union[np.ndarray, list], transform_method: Literal['pad_sequences'] = 'pad_sequences', **transform_kwargs) -> Union[torch.tensor, list]:
        """
        Converts the data to a tensor if the data is a numpy array.
        Otherwise, when the data is a list, it calls a transform method to convert this list to a single pytorch tensor.
        By default, this transformation method will padd 0 to the sequences to make them of the same length.
        """
        if isinstance(data, np.ndarray):
            return torch.tensor(data)
        elif isinstance(data, list):
            return self.convert_list_of_arrays_to_tensor(data, transform_method, **transform_kwargs)
        else:
            raise ValueError(f'Cannot convert data of type {type(data)} to a tensor')
        
    def convert_dict_to_dict_of_tensors(self, data: dict) -> dict:
        """
        Converts the data dictionary to a dictionary of tensors
        """
        output_dict = {}
        for key in data:
            output_dict[key] = self.convert_to_tensor(data[key])
        return output_dict
    
    def convert_list_of_arrays_to_tensor(self, data: list, transform_method: str, **transform_kwargs) -> torch.tensor:
        """
        convert a list of arrays of variable sizes to a single torch tensor
        """
        return self.__getattribute__(transform_method)(data, **transform_kwargs)
    
    def pad_sequences(self, data: list, **transform_kwargs) -> torch.tensor:
        """
        Pads the sequences in the data with a value
        kwargs are padding_value and batch_first, see pad_sequence documentation in pytorch for more information
        """
        batch_first = transform_kwargs.get('batch_first', True)
        padding_value = transform_kwargs.get('padding_value', 0)
        # convert each element of data to a torch tensor
        data = [torch.tensor(item) for item in data]
        return pad_sequence(data, batch_first=batch_first, padding_value=padding_value)
    
    def get_dictionary_per_idx(self, dictionary: dict, idx: int) -> dict:
        """
        Get the dictionary for a specific index
        """
        return {key: dictionary[key][idx] for key in dictionary}

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[dict, dict, dict]:
        return self.get_dictionary_per_idx(self.input, idx), self.get_dictionary_per_idx(self.label, idx), self.get_dictionary_per_idx(self.meta, idx)
