"""
This file contains the parser class for parsing an input CSV file which is the STIMULUS data format.

The file contains a header column row where column names are formated as is : 
name:category:type

name is straightforward, it is the name of the column
category corresponds to any of those three values : input, meta, or label. Input is the input of the deep learning model, label is the output (what needs to be predicted) and meta corresponds to metadata not used during training (could be used for splitting).
type corresponds to the data type of the columns, as specified in the types module. 

The parser is a class that takes as input a CSV file and a experiment class that defines data types to be used, noising procedures, splitting etc. 
"""

import pandas as pd
from typing import Any, Tuple

#TODO create a base class for CsvParser and CsvHandler

class CSVParser: # change to CsvHandler
    """
    Class for parsing CSV files.
    
    It will parse the CSV file into three dictionaries, one for each category [input, label, meta].
    So each dictionary will have the keys in the form name:type, and the values will be the column values.
    Then, one can get one or many items from the data, encoded.
    """
    
    def __init__(self, experiment: Any, csv_path: str) -> None:
        self.experiment = experiment
        self.csv_path = csv_path
        self.input, self.label, self.meta = self.parse_csv_to_input_label_meta(self.csv_path)
        self.padding_value = self.find_padding_value(self.input)
        
    def parse_csv_to_input_label_meta(self, csv_path: str) -> Tuple[dict, dict, dict]:
        """
        This function reads the csv file into a dictionary, 
        and then parses each key with the form name:category:type 
        into three dictionaries, one for each category [input, label, meta].
        The keys of each new dictionary are in this form name:type.
        """
        # read csv file into a dictionary of lists
        # the keys of the dictionary are the column names and the values are the column values
        data = pd.read_csv(csv_path, dtype=str).to_dict(orient="list")
        
        # parse the dictionary into three dictionaries, one for each category [input, label, meta]
        input_data, label_data, meta_data = {}, {}, {}
        for key in data:
            name, category, data_type = key.split(":")
            if category.lower() == "input":
                input_data[f"{name}:{data_type}"] = data[key]
            elif category.lower() == "label":
                label_data[f"{name}:{data_type}"] = data[key]
            elif category.lower() == "meta":
                meta_data[f"{name}:{data_type}"] = data[key]
            else:
                raise ValueError(f"Unknown category {category}, category (the second element of the csv column, seperated by ':') should be input, label or meta. The specified csv column is {key}.")
        return input_data, label_data, meta_data

    def find_padding_value(self, data: dict) -> int:
        """
        Find an integer that is not present in any of the lists of the data dictionary
        """
        i = 0
        while True:
            if i not in [item for sublist in data.values() for item in sublist]:
                return i
            i += 1
    
    def get_and_encode(self, dictionary: dict, idx: Any) -> dict:
        """
        It gets the data at a given index, and encodes it according to the data_type.

        `dictionary`:
            The keys of the dictionaries are always in the form `name:type`.
            `type` should always match the name of the initialized data_types in the Experiment class. So if there is a `dna` data_type in the Experiment class, then the input key should be `name:dna`
        `idx`:
            The index of the data to be returned, it can be a single index, a list of indexes or a slice

        The return value is a dictionary containing numpy array of the encoded data at the given index.
        """
        output = {}
        for key in dictionary: # processing each column
            
            # get the name and data_type
            name = key.split(":")[0]
            data_type = key.split(":")[1]

            # get the data at the given index
            # if the data is not a list, it is converted to a list
            # otherwise it breaks Float().encode_all(data) because it expects a list
            data = dictionary[key][idx]
            if not isinstance(data, list):
                data = [data]

            # check if 'data_type' is in the experiment class attributes
            if not hasattr(self.experiment, data_type.lower()):
                raise ValueError(f"The data type {data_type} is not in the experiment class attributes. the column name is {key}, the available attributes are {self.experiment.__dict__}")
            
            # encode the data at given index
            # For that, it first retrieves the data object and then calls the encode_all method to encode the data

            
            output[name] = self.experiment.get_encoding_all(data_type)(dictionary[key][idx])

    
        return output
    
    def get_encoded_item(self, idx: Any) -> Tuple[dict, dict, dict]:
        """
        It gets the data at a given index, and encodes the input and label, leaving meta as it is.
        """
        x = self.get_and_encode(self.input, idx)
        y = self.get_and_encode(self.label, idx)
        return x, y, self.meta
    
    def __len__(self) -> int:
        """
        returns the length of the first list in input, assumes that all are the same length
        """
        return len(list(self.input.values())[0])
    
    def __getitem__(self, idx: Any) -> dict:
        """
        get a dictionary with all the keys for the data at a given index
        """
        data = {**self.input, **self.label, **self.meta}
        return { key: data[key][idx] for key in data }
    