"""
This file contains the parser class for parsing an input CSV file which is the STIMULUS data format.

The file contains a header column row where column names are formated as is :
name:category:type

name is straightforward, it is the name of the column
category corresponds to any of those three values : input, meta, or label. Input is the input of the deep learning model, label is the output (what needs to be predicted) and meta corresponds to metadata not used during training (could be used for splitting).
type corresponds to the data type of the columns, as specified in the types module.

The parser is a class that takes as input a CSV file and a experiment class that defines data types to be used, noising procedures, splitting etc.
"""

import numpy as np
import polars as pl
from typing import Any, Tuple, Union
from functools import partial

class CsvHandler:
    """
    Meta class for handling CSV files.
    """
    def __init__(self, experiment: Any, csv_path: str) -> None:
        self.experiment = experiment
        self.csv_path = csv_path
        self.categories = self.check_and_get_categories()
        self.check_compulsory_categories_exist()

    def check_and_get_categories(self) -> list:
        """
        Returns the categories contained in the csv file.
        """
        with open(self.csv_path, 'r') as f:
            header = f.readline().strip().split(',')
        categories = []
        for colname in header:
            category = colname.split(":")[1].lower()
            if category not in ['input', 'label', 'split', 'meta']:
                raise ValueError(f"Unknown category {category}, category (the second element of the csv column, seperated by ':') should be input, label, split or meta. The specified csv column is {colname}.")
            categories.append(category)
        return categories

    def update_categories(self) -> None:
        """
        Updates the categories of the csv file.
        Checks colnames in header and updates the categories that are present.
        """
        for colname in self.data.columns:
            category = colname.split(":")[1].lower()
            if category not in self.categories:
                self.categories.append(category)
        

    def extract_header(self) -> list:
        """
        Extracts the header of the csv file.
        """
        with open(self.csv_path, 'r') as f:
            header = f.readline().strip().split(',')
        return header
    
    def get_keys_from_header(self, header, column_name: str = None, category: str = None, data_type: str = None) -> list:
        keys = []
        for key in header:
            current_name, current_category, current_dtype = key.split(":")
            if (column_name is None or column_name == current_name) and (category is None or category == current_category) and (data_type is None or data_type == current_dtype):
                keys.append(key)
        if len(keys) == 0:
            raise ValueError(f"No keys found with the specified column_name={column_name}, category={category}, data_type={data_type}")
        return keys   
        

    def get_keys_based_on_name_category_dtype(self, column_name: str = None, category: str = None, data_type: str = None) -> list:
        """
        Returns the keys that are of a specific type, name or category. Or a combination of those.
        """
        if (column_name is None) and (category is None) and (data_type is None):
            raise ValueError(f"At least one of the arguments column_name, category or data_type should be provided")
        header = self.extract_header()
        keys = self.get_keys_from_header(header, column_name, category, data_type)
        return keys


    def check_compulsory_categories_exist(self) -> None:
        """
        Checks if the compulsory categories exist in the csv file.
        """
        if 'input' not in self.categories:
            raise ValueError(f"The category input is not present in the csv file")

    def load_csv(self) -> pl.DataFrame:
        """
        Loads the csv file into a polars dataframe.
        """
        return pl.read_csv(self.csv_path)
            

class CsvProcessing(CsvHandler):
    """
    Class to load the input csv data and add noise accordingly.
    """
    def __init__(self, experiment: Any, csv_path: str) -> None:
        super().__init__(experiment, csv_path)
        self.data = self.load_csv()

    def add_split(self, config: dict,  force=False) -> None:
        """
        Add a column specifying the train, validation, test splits of the data.
        An error exception is raised if the split column is already present in the csv file. This behaviour can be overriden by setting force=True.

        args:
            config (dict) : the dictionary containing  the following keys:
                            "name" (str)        : the split_function name, as defined in the splitters class and experiment.
                            "parameters" (dict) : the split_function specific optional parameters, passed here as a dict with keys named as in the split function definition.
            force (bool) : If True, the split column will be added even if it is already present in the csv file.
        """
        if ('split' in self.categories) and (not force):
            raise ValueError(f"The category split is already present in the csv file. If you want to still use this function, set force=True")
        
        # set the split name method
        split_method = config["name"]

        # get the indices for train, validation and test using the specified split method
        train, validation, test = self.experiment.get_function_split(split_method)(self.data, **config['params'])

        # add the split column to the data
        split_column = np.full(len(self.data), -1).astype(int)
        split_column[train] = 0
        split_column[validation] = 1
        split_column[test] = 2
        self.data = self.data.with_columns(pl.Series('split:split:int', split_column))
        self.update_categories()
        

    def transform(self, transformations: list) -> None:
        """
        Transforms the data using the specified configuration.
        """
        for dictionary in transformations:
            key = dictionary['column_name']
            data_type = key.split(':')[2]
            data_transformer = dictionary['name']
            transfomer = self.experiment.get_data_transformer(data_type, data_transformer)
            
            # If the transformer is only for training data, we need to separate the data
            # and transform only the training data
            if transfomer.training_data_only:
                split_colname = self.get_keys_from_header(self.data.columns, category='split')
                data_to_transform = self.data.filter(pl.col(split_colname) == 0)
                untransformed_data = self.data.filter(pl.col(split_colname) != 0)
            else: 
                data_to_transform = self.data
            
            # Transform the data
            new_data = transfomer.transform_all(list(data_to_transform[key]), **dictionary['params'])
            
            # Add the transformed data to the dataframe
            
            # If the transformer modifies the column, we need to replace the column
            if transfomer.add_row:
                new_rows = data_to_transform.with_columns(pl.Series(key, new_data))
                self.data = self.data.vstack(new_rows)
            else:              
                transformed_data = data_to_transform.with_columns(pl.Series(key, new_data))
                # make sure the column has the same type as the new data
                # this is necessary because the transformer could change the type of the column (e.g. from int to float) 
                transformed_data_type = str(transformed_data[key].dtype)
                untransformed_data = untransformed_data.with_columns(pl.col(key).cast(getattr(pl, transformed_data_type)))
 
                # If the transformer is only for training data, we need to concatenate the transformed data with the untransformed data
                if transfomer.training_data_only:
                    self.data = transformed_data.vstack(untransformed_data)
                else:
                    self.data = transformed_data

                                 
    def shuffle_labels(self) -> None:
        """
        Shuffles the labels in the data.
        """
        label_keys = self.get_keys_based_on_name_category_dtype(category='label')
        for key in label_keys:
            self.data = self.data.with_columns(pl.Series(key, np.random.permutation(list(self.data[key]))))

    def save(self, path: str) -> None:
        """
        Saves the data to a csv file.
        """
        self.data.write_csv(path)

class CsvLoader(CsvHandler):
    """
    Class for loading the csv data, and then encode the information.

    It will parse the CSV file into four dictionaries, one for each category [input, label, meta].
    So each dictionary will have the keys in the form name:type, and the values will be the column values.
    Afterwards, one can get one or many items from the data, encoded.
    """
    def __init__(self, experiment: Any, csv_path: str, split: Union[int, None] = None) -> None:
        """
        Initialize the class by parsing and splitting the csv data into the corresponding categories.

        args:
            experiment (class) : The experiment class to perform
            csv_path (str) : The path to the csv file
            split (int) : The split to load, 0 is train, 1 is validation, 2 is test.
        """
        super().__init__(experiment, csv_path)

        # we need a different parsing function in case we have the split argument or not
        # NOTE using partial we can define the default split value, without the need to pass it as an argument all the time through the class
        if split is not None:
            prefered_load_method = partial(self.load_csv_per_split, split=split)
        else:
            prefered_load_method = self.load_csv

        # parse csv and split into categories
        self.input, self.label, self.meta = self.parse_csv_to_input_label_meta(prefered_load_method)

    def load_csv_per_split(self, split: int) -> pl.DataFrame:
        """
        Load the part of csv file that has the specified split value.
        Split is a number that for 0 is train, 1 is validation, 2 is test.
        This is accessed through the column with category `split`. Example column name could be `split:split:int`.

        NOTE that the aim of having this function is that depending on the training, validation and test scenarios,
        we are gonna load only the relevant data for it.
        """
        if 'split' not in self.categories:
            raise ValueError(f"The category split is not present in the csv file")
        if split not in [0, 1, 2]:
            raise ValueError(f"The split value should be 0, 1 or 2. The specified split value is {split}")
        colname = self.get_keys_based_on_name_category_dtype("split")
        if len(colname) > 1:
            raise ValueError(f"The split category should have only one column, the specified csv file has {len(colname)} columns")
        colname = colname[0]
        return pl.scan_csv(self.csv_path).filter(pl.col(colname) == split).collect()

    def parse_csv_to_input_label_meta(self, load_method: Any) -> Tuple[dict, dict, dict]:
        """
        This function reads the csv file into a dictionary,
        and then parses each key with the form name:category:type
        into three dictionaries, one for each category [input, label, meta].
        The keys of each new dictionary are in this form name:type.
        """
        # read csv file into a dictionary of lists
        # the keys of the dictionary are the column names and the values are the column values
        data = load_method().to_dict(as_series=False)

        # parse the dictionary into three dictionaries, one for each category [input, label, meta]
        input_data, label_data, split_data, meta_data = {}, {}, {}, {}
        for key in data:
            name, category, data_type = key.split(":")
            if category.lower() == "input":
                input_data[f"{name}:{data_type}"] = data[key]
            elif category.lower() == "label":
                label_data[f"{name}:{data_type}"] = data[key]
            elif category.lower() == "meta":
                meta_data[f"{name}"] = data[key]
        return input_data, label_data, meta_data

    def get_and_encode(self, dictionary: dict, idx: Any = None) -> dict:
        """
        It gets the data at a given index, and encodes it according to the data_type.

        `dictionary`:
            The keys of the dictionaries are always in the form `name:type`.
            `type` should always match the name of the initialized data_types in the Experiment class. So if there is a `dna` data_type in the Experiment class, then the input key should be `name:dna`
        `idx`:
            The index of the data to be returned, it can be a single index, a list of indexes or a slice
            If None, then it encodes for all the data, not only the given index or indexes.

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
            data = dictionary[key] if idx is None else dictionary[key][idx]
            
            if not isinstance(data, list):
                data = [data]

            # check if 'data_type' is in the experiment class attributes
            if not hasattr(self.experiment, data_type.lower()):
                raise ValueError("The data type", data_type, "is not in the experiment class attributes. the column name is", key, "the available attributes are", self.experiment.__dict__)

            # encode the data at given index
            # For that, it first retrieves the data object and then calls the encode_all method to encode the data
            output[name] = self.experiment.get_function_encode_all(data_type)(data)

        return output

    def get_all_items(self) -> Tuple[dict, dict, dict]:
        """
        Returns all the items in the csv file, encoded.
        TODO in the future we can optimize this for big datasets (ie. using batches, etc).
        """
        return self.get_and_encode(self.input), self.get_and_encode(self.label), self.meta

    def get_all_items_and_length(self) -> Tuple[dict, dict, dict, int]:
        """
        Returns all the items in the csv file, encoded, and the length of the data.
        """
        return self.get_and_encode(self.input), self.get_and_encode(self.label), self.meta, len(self)

    def __len__(self) -> int:
        """
        returns the length of the first list in input, assumes that all are the same length
        """
        return len(list(self.input.values())[0])

    def __getitem__(self, idx: Any) -> dict:
        """
        It gets the data at a given index, and encodes the input and label, leaving meta as it is.

        `idx`:
            The index of the data to be returned, it can be a single index, a list of indexes or a slice
        """
        # encode input and labels for given index
        x = self.get_and_encode(self.input, idx)
        y = self.get_and_encode(self.label, idx)

        # get the meta data at the given index for each key
        meta = {}
        for key in self.meta:
            data = self.meta[key][idx]
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            meta[key] = data

        return x, y, meta
