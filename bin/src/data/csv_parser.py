"""
This file contains the parser class for parsing an input CSV file which is the STIMULUS data format.

The file contains a header column row where column names are formated as is : 
name:type:category

name is straightforward, it is the name of the column
category corresponds to any of those three values : input, meta, or label. Input is the input of the deep learning model, label is the output (what needs to be predicted) and meta corresponds to metadata not used during training (could be used for splitting).
type corresponds to the data type of the columns, as specified in the types module. 

The parser is a class that takes as input a CSV file and a experiment class that defines data types to be used, noising procedures, splitting etc. 

"""

import csv
from typing import Any, Tuple


class CSVParser:
    """
    Class for parsing CSV files.
    """
    
    def __init__(self, experiment: Any, csv_path: str) -> None:
        self.experiment = experiment
        self.csv_path = csv_path
        self.data = self.parse_csv(csv_path)
        self.input, self.label, self.meta = self.split_input_label_meta(self.data)

    def parse_csv(self, csv_path:str) -> dict:
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            data = {}
            for row in reader:
                for i in range(len(row)):
                    if header[i] not in data:
                        data[header[i]] = []
                    data[header[i]].append(row[i])
            return data
    
    def split_input_label_meta(self, data: dict) -> Tuple[dict, dict, dict]:
        """
        The dict data has keys of this form : name:category:type . The category corresponds to input, label or meta. 
        This function splits the data into three dictionaries accordingly, one for each category. 
        The keys of each new dictionary are in this form name:type.
        """
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
    
    def get_and_encode(self, dictionary: dict, idx: Any) -> dict:
        """
        Returns the data from a dictionary at a given index. 
        The keys of the dictionaries are always in the form name:type.
        The type should always matches the name of the initialized data_types in the Experiment class. 
        The idx is the index of the data to be returned, it can be a single index, a list of indexes or a splice

        The return value is a dictionary containing numpy array of the data at the given index, and for the input and output, should be encoded using the encode_all method of the relevant data_type class
        """
        output = {}
        for key in dictionary: # processing each column
            name = key.split(":")[0] # each column is formated as name:type:category
            data_type = key.split(":")[1]
            # check if 'data_type' is in the experiment class attributes
            if not hasattr(self.experiment, data_type.lower()):
                raise ValueError(f"The data type {data_type} is not in the experiment class attributes. the column name is {key}, the available attributes are {self.experiment.__dict__}")
            
            # encode the data using the encode_all method of the relevant data_type class
            output[name] = self.experiment.__getattribute__(data_type.lower()).encode_all(dictionary[key][idx])
    
        return output
    
    def get_encoded_item(self, idx: Any) -> Tuple[dict, dict, dict]:
        x = self.get_and_encode(self.input, idx)
        y = self.get_and_encode(self.label, idx)
        return x, y, self.meta
    
    def __len__(self) -> int:
        return len(self.input)
    
    def __getitem__(self, idx: Any) -> dict:
        """
        get a dictionary from self.data at a given index, knowing that self.data is a dictionary of lists
        """
        return { key: self.data[key][idx] for key in self.data}


    