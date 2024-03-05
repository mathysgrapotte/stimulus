"""
This file provides the class API for handling the data in pytorch using the Dataset and Dataloader classes
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from csv_parser import CSVParser
from typing import Any, Tuple

class TorchDataset(Dataset):
    """
    Class for creating a torch dataset
    """
    def __init__(self, csvpath : str, experiment : Any) -> None:
        self.csvpath = csvpath
        self.parser = CSVParser(experiment, csvpath)

    def convert_list_of_numpy_arrays_to_tensor(self, data: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a list of numpy arrays to a tensor. 
        If the list includes numpy arrays of different shapes, padd the numpy arrays and return a mask tensor, otherwise mask tensor is set to None

        # TODO write an improvement where the padding is done with a specific integer and the mask is built from that padding value for performance.
        """
        if len(data) > 1:
            # check if the data is of different shapes
            if len(set([d.shape for d in data])) == 1:
                return torch.tensor(data), None
            
            # otherwise, pad the data and build a mask tensor that points to where the data has been padded.
            else:
                max_shape = max([d.shape for d in data])
                mask = torch.zeros(len(data), max_shape[0], max_shape[1])
                for i, d in enumerate(data):
                    mask[i, :d.shape[0], :d.shape[1]] = 1
                return pad_sequence(data), mask

        else:
            return torch.tensor(data[0]), None

    def convert_dict_to_tensor(self, data: dict) -> dict:
        """
        Converts the data in a dictionary at all keys to a torch tensor, assuming the data is convertible to a tensor.
        """
        output_dict = {}
        mask_dict = {}
        for key in data:
            output_dict[key], mask_dict[key] = self.convert_list_of_numpy_arrays_to_tensor(data[key])
        return output_dict, mask_dict

    def __len__(self) -> int:
        return len(self.parser)

    def __getitem__(self, idx: int) -> Tuple[dict, dict, dict]:
        x, y, meta = self.parser.get_encoded_item(idx)
        # convert the content in the x and y directories to torch tensors
        x, x_mask = self.convert_to_tensor(x)
        y, y_mask = self.convert_to_tensor(y)
        return x, y, x_mask, y_mask, meta


