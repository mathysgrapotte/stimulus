import json
import os
import unittest
import sys
import numpy as np
import numpy.testing as npt
import polars as pl
sys.path.append('./')
from bin.src.data.csv import CsvProcessing, CsvLoader
from bin.src.data.experiments import DnaToFloatExperiment,ProtDnaToFloatExperiment
from bin.src.data.experiments import DnaToFloatExperiment, ProtDnaToFloatExperiment

class AbstractTestCsvProcessing(unittest.TestCase):
    """
    Abstract class for testing CsvProcessing class.
    """
    def _test_len(self):
        """
        It tests that it loads correctly the data with the correct shape.
        """
        self.assertEqual(len(self.csv_processing.data), self.data_length)

    def _add_split(self):
        self.csv_processing.add_split(self.configs['split'])
    
    def _test_random_splitter(self, expected_splits):
        """
        It tests that the data is split correctly.
        """
        for i in range(self.data_length):
            col = self.csv_processing.data['split:split:int'][i]
            self.assertEqual(self.csv_processing.data['split:split:int'][i], expected_splits[i])

    def _transform(self):
        self.csv_processing.transform(self.configs['transform'])

    def _test_first_value_from_column(self, column_name, expected_value, position=0):
        """
        It tests the first value of a specific column.
        """
        observed_value = list(self.csv_processing.data[column_name])[position]
        if isinstance(observed_value, float):
            observed_value = round(observed_value, 6)
        self.assertEqual(observed_value, expected_value)
    
    def _test_all_values_in_column(self, column_name, expected_values):
        """
        It tests all the values of a specific column.
        """
        observed_values = list(self.csv_processing.data[column_name])
        for i in range(len(observed_values)):
            if isinstance(observed_values[i], float):
                observed_values[i] = round(observed_values[i], 6)
        self.assertEqual(observed_values, expected_values)

class TestDnaToFloatCsvProcessing(AbstractTestCsvProcessing):
    """
    Test CsvProcessing class for DnaToFloatExperiment
    """
    def setUp(self):
        np.random.seed(123)
        pl.set_random_seed(123)
        self.experiment = DnaToFloatExperiment()
        self.csv_path = os.path.abspath("bin/tests/test_data/dna_experiment/test.csv")
        self.csv_processing = CsvProcessing(self.experiment, self.csv_path)
        self.csv_shuffle_long_path = os.path.abspath("bin/tests/test_data/dna_experiment/test_shuffling_long.csv")
        self.csv_shuffle_long = CsvProcessing(self.experiment, self.csv_shuffle_long_path)
        self.csv_shuffle_long_shuffled_path = os.path.abspath("bin/tests/test_data/dna_experiment/test_shuffling_long_shuffled.csv")
        self.csv_shuffle_long_shuffled = CsvProcessing(self.experiment, self.csv_shuffle_long_shuffled_path)
        with open('bin/tests/test_data/dna_experiment/test_config.json', 'rb') as f:
            self.configs = json.load(f)
        self.data_length = 2

    def test_len(self):
        self._test_len()

    def test_split_and_noise(self):
        self._test_first_value_from_column('hello:input:dna', 'ACTGACTGATCGATGC')
        self._test_first_value_from_column('hola:label:float', 12)
        self._add_split()
        self._test_random_splitter([1, 0])
        self._transform()
        self.data_length = self.data_length * 2
        self._test_len()
        self._test_all_values_in_column('pet:meta:str', ['dog', 'cat', 'dog','cat'])
        self._test_all_values_in_column('hola:label:float', [12.676405, 12.0, 12.676405, 12.0])
        self._test_all_values_in_column('hello:input:dna', ['ACTGACTGATCGATNN', 'ACTGACTGATCGATGC', 'NNATCGATCAGTCAGT', 'GCATCGATCAGTCAGT'])
        self._test_all_values_in_column('split:split:int', [0, 1, 0, 1])
        
    def test_shuffle_labels(self):
        # initialize seed to 42 to make the test reproducible
        np.random.seed(42)
        self.csv_shuffle_long.shuffle_labels()
        npt.assert_array_equal(self.csv_shuffle_long.data['hola:label:float'], self.csv_shuffle_long_shuffled.data['hola:label:float'])



class TestProtDnaToFloatCsvProcessing(AbstractTestCsvProcessing):
    """
    Test CsvProcessing class for ProtDnaToFloatExperiment
    """
    def setUp(self):
        self.experiment = ProtDnaToFloatExperiment()
        self.csv_path = os.path.abspath("bin/tests/test_data/prot_dna_experiment/test.csv")
        self.csv_processing = CsvProcessing(self.experiment, self.csv_path)
        with open('bin/tests/test_data/prot_dna_experiment/test_config.json', 'rb') as f:
            self.configs = json.load(f)
        self.data_length = 2

    def test_len(self):
        self._test_len()

    def test_split_and_noise(self):
        self._test_first_value_from_column('bonjour:input:prot', 'GPRTTIKAKQLETLK')
        self._test_first_value_from_column('hello:input:dna', 'ACTGACTGATCGATGC')
        self._test_first_value_from_column('hola:label:float', 12)
        self._add_split()
        self._test_random_splitter([1, 0])
        self._transform()
        self.data_length = self.data_length * 2
        self._test_len()
        self._test_all_values_in_column('pet:meta:str', ['dog', 'cat', 'dog','cat'])
        self._test_all_values_in_column('hola:label:float', [12.676405, 12.0, 12.676405, 12.0])
        self._test_all_values_in_column('hello:input:dna', ['ACTGACTGATCGATNN', 'ACTGACTGATCGATGC', 'NNATCGATCAGTCAGT', 'GCATCGATCAGTCAGT'])
        self._test_all_values_in_column('split:split:int', [0, 1, 0,1])
        self._test_all_values_in_column('bonjour:input:prot', ['GPRTTIKAKQLETLX', 'GPRTTIKAKQLETLK', 'GPRTTIKAKQLETLX', 'GPRTTIKAKQLETLK'])

class AbstractTestCsvLoader(unittest.TestCase):
    """
    Abstract class for testing CsvLoader class.
    """
    def _test_len(self):
        """
        It tests the length of the dataset.
        """
        self.assertEqual(len(self.csv_loader), self.data_shape[0])

    def _test_parse_csv_to_input_label_meta(self):
        """
        It tests that the csv is parsed to input, label and meta.
        """
        self.assertIsInstance(self.csv_loader.input, dict)
        self.assertIsInstance(self.csv_loader.label, dict)
        self.assertIsInstance(self.csv_loader.meta, dict)

    def _test_get_encoded_item_unique(self):
        """
        It tests that the csv_loader.get_encoded_item works well when getting one item.
        """
        # get the encoded item from the csv file at idx 0
        encoded_item = self.csv_loader[0]

        # test that the encoded item is a tuple of three dictionaries [input, label, meta]
        # also each element inside a dictionary is a np array of length 1
        self.assertEqual(len(encoded_item), 3)
        for i in range(3):
            self.assertIsInstance(encoded_item[i], dict)
            for key in encoded_item[i].keys():
                self.assertIsInstance(encoded_item[i][key], np.ndarray)
                try:
                    self.assertEqual(len(encoded_item[i][key]), 1)
                except TypeError:
                    # scalars do not have a length and return a TypeError if len() is called on them
                    self.assertEqual(encoded_item[i][key].size, 1)

    def _test_get_encoded_item_multiple(self):
        """
        It tests that the csv_loader.get_encoded_item works well when getting multiple items using slice.
        """
        # get the encoded items from the csv file at idx 0 and 1
        encoded_item = self.csv_loader[slice(0, 2)]

        # test that the encoded item is a tuple of three dictionaries [input, label, meta]
        # also each element inside a dictionary is a list of length 2
        self.assertEqual(len(encoded_item), 3)
        for i in range(3):
            self.assertIsInstance(encoded_item[i], dict)
            for key in encoded_item[i].keys():
                self.assertIsInstance(encoded_item[i][key], np.ndarray)
                self.assertEqual(len(encoded_item[i][key]), 2)

    def _test_load_with_split(self):
        """
        Test that the csv_loader works well when split is provided.
        """
        # test when split is not provided
        self.csv_loader_split = CsvLoader(self.experiment, self.csv_path_split)
        self.assertEqual(len(self.csv_loader_split), self.data_shape_split[0])

        # test when split is 0, 1 or 2
        for i in [0, 1, 2]:
            self.csv_loader_split = CsvLoader(self.experiment, self.csv_path_split, split=i)
            self.assertEqual(len(self.csv_loader_split.input['hello:dna']), self.shape_splits[i])

        # test when split is not valid
        with self.assertRaises(ValueError): # should raise an error
            self.csv_loader_split = CsvLoader(self.experiment, self.csv_path_split, split=3)

    def _test_get_all_items(self):
        """
        Test that the csv_loader.get_all_items works well.
        """
        input_data, label_data, meta_data = self.csv_loader.get_all_items()
        self.assertIsInstance(input_data, dict)
        self.assertIsInstance(label_data, dict)
        self.assertIsInstance(meta_data, dict)


class TestDnaToFloatCsvLoader(AbstractTestCsvLoader):
    """
    Test CsvLoader class for DnaToFloatExperiment
    """
    def setUp(self):
        self.csv_path = os.path.abspath("bin/tests/test_data/dna_experiment/test.csv")
        self.csv_path_split = os.path.abspath("bin/tests/test_data/dna_experiment/test_with_split.csv")
        self.experiment = DnaToFloatExperiment()
        self.csv_loader = CsvLoader(self.experiment, self.csv_path)
        self.data_shape = [2,3]
        self.data_shape_split = [48,4]
        self.shape_splits = {0: 16, 1: 16, 2: 16}

    def test_len(self):
        self._test_len()

    def test_parse_csv_to_input_label_meta(self):
        self._test_parse_csv_to_input_label_meta()

    def test_get_encoded_item_unique(self):
        self._test_get_encoded_item_unique()

    def test_get_encoded_item_multiple(self):
        self._test_get_encoded_item_multiple()

    def test_load_with_split(self):
        self._test_load_with_split()

    def test_get_all_items(self):
        self._test_get_all_items()

class TestProtDnaToFloatCsvLoader(AbstractTestCsvLoader):
    """
    Test CsvLoader class for ProtDnaToFloatExperiment
    """
    def setUp(self):
        self.csv_path = os.path.abspath("bin/tests/test_data/prot_dna_experiment/test.csv")
        self.csv_path_split = os.path.abspath("bin/tests/test_data/prot_dna_experiment/test_with_split.csv")
        self.experiment = ProtDnaToFloatExperiment()
        self.csv_loader = CsvLoader(self.experiment, self.csv_path)
        self.data_shape = [2,4]
        self.data_shape_split = [3,5]
        self.shape_splits = {0: 1, 1: 1, 2: 1}

    def test_len(self):
        self._test_len()

    def test_parse_csv_to_input_label_meta(self):
        self._test_parse_csv_to_input_label_meta()

    def test_get_encoded_item_unique(self):
        self._test_get_encoded_item_unique()

    def test_get_encoded_item_multiple(self):
        self._test_get_encoded_item_multiple()

    def test_load_with_split(self):
        self._test_load_with_split()

    def test_get_all_items(self):
        self._test_get_all_items()


if __name__ == "__main__":
    unittest.main()