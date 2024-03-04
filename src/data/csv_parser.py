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
import os
from typing import Any 
from experiments import *


class CSVParser:
    """
    Class for parsing CSV files.
    """
    
    def __init__(self, experiment: Any, csv_path: str) -> None:
        self.experiment = experiment
        self.csv_path = csv_path
        self.data = self.parse_csv(csv_path)

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

if __name__ == "__main__":
    parser = CSVParser(None, os.path.abspath("../../test_data/test.csv"))
    data = parser.parse_csv()
    print(data)
    print("hello world!")