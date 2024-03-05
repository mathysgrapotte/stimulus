"""
This file provides the class API for handling the data in pytorch using the Dataset and Dataloader classes
"""

import torch
import experiments as exp
from torch.utils.data import Dataset, DataLoader
from csv_parser import CSVParser
from typing import Any, Tuple

class TorchDataset(Dataset):
    """
    Class for creating a torch dataset
    """

    def convert_to_tensor(self, data: dict) -> dict:
        """
        Converts the data in a dictionary at all keys to a torch tensor, assuming the data is convertible to a tensor.
        """
        for key in data:
            data[key] = torch.tensor(data[key])
        return data

    def __init__(self, csvpath : str, experiment : Any) -> None:
        self.csvpath = csvpath
        self.parser = CSVParser(experiment, csvpath)

    def __len__(self) -> int:
        return len(self.parser)

    def __getitem__(self, idx: int) -> Tuple[dict, dict, dict]:
        x, y, meta = self.parser.get_encoded_item(idx)
        # convert the content in the x and y directories to torch tensors
        x = self.convert_to_tensor(x)
        y = self.convert_to_tensor(y)
        return x, y, meta


