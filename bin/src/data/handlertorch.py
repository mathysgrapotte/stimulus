"""
This file provides the class API for handling the data in pytorch using the Dataset and Dataloader classes
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .csv import CsvLoader
from typing import Any, Tuple

class TorchDataset(Dataset):
    """
    Class for creating a torch dataset
    """
    def __init__(self, csvpath: str, experiment: Any, split: Tuple[None, int] = None) -> None:
        self.csvpath = csvpath
        self.parser = CsvLoader(experiment, csvpath, split=split)

    def convert_list_of_numpy_arrays_to_tensor(self, data: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a list of numpy arrays to a tensor. 
        If the list includes numpy arrays of different shapes, padd the numpy arrays and return a mask tensor, otherwise mask tensor is set to None

        # TODO: This method utilizes ifs to check the shape of the data. this is not ideal. Performance improvement could be done here.
        # TODO: possibly we want to convert everything at once on load and not on the fly.
        """
        if len(data) > 1:
            # check if data is a flat list (of float or integers):
            if isinstance(data[0], (float, int)):
                return torch.tensor(data), {}

            # check if the data is of different shapes
            elif len(set([d.shape for d in data])) == 1:
                return torch.tensor(np.array(data)), {}
            
            # otherwise, pad the data and build a mask tensor that points to where the data has been padded.
            else:
                data = [torch.from_numpy(d) for d in data] # convert the np arrays to tensors

                # pad sequences
                padded_data = pad_sequence(data, batch_first=True, padding_value=42)

                # create a mask of the same shape as the padded data
                mask = torch.zeros_like(padded_data)

                # mask should have ones everywhere the data is not padded (so values are not 42)
                mask[padded_data != 42] = 1

                return padded_data, mask

        else:
            return torch.tensor(data[0]), {}

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
        x, y, meta = self.parser[idx]
        # convert the content in the x and y directories to torch tensors
        x, x_mask = self.convert_dict_to_tensor(x)
        y, y_mask = self.convert_dict_to_tensor(y)
        return x, y, x_mask, y_mask, meta


