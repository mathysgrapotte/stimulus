import json
import os
import unittest
from bin.src.data.csv import CsvProcessing, CsvLoader
from bin.src.data.experiments import ProtDnaToFloatExperiment


class TestProtDnaToFloatCsvProcessing(unittest.TestCase):

    def setUp(self):
        self.csv_path = os.path.abspath("bin/tests/test_data/test.csv")
        with open('bin/tests/test_data/test_config.json', 'rb') as f:
            self.configs = json.load(f)

    def test_load_csv(self):
        """
        It tests that it can load the csv file correctly.
        """
        csv_processing = CsvProcessing(ProtDnaToFloatExperiment(), self.csv_path)
        self.assertEqual(csv_processing.data.shape[0], 2)
        self.assertEqual(csv_processing.data.shape[1], 3)

    def test_add_noise(self):
        """
        It tests the add_noise method.
        """
        # load data and add noise
        csv_processing = CsvProcessing(ProtDnaToFloatExperiment(), self.csv_path)
        csv_processing.add_noise(self.configs['noise'])

        # check protein mask
        self.assertEqual(list(csv_processing.data['bonjour:input:prot'])[0], 'GPRTTIKAKQLETLX')

        # check dna mask
        self.assertEqual(list(csv_processing.data['hello:input:dna'])[0], 'ACTGACTGATCGATNN')

        # check float gaussian noise
        self.assertEqual(12.68, round(list(csv_processing.data['hola:label:float'])[0],2))


class TestDnaToFloatCsvLoader(unittest.TestCase):

    def setUp(self):
        self.csv_loader = CsvLoader(ProtDnaToFloatExperiment(), os.path.abspath("bin/tests/test_data/test.csv"))

    def test_get_encoded_item_unique(self):
        """ 
        It tests that the csv_loader.get_encoded_item works well when getting one item.
        The following test is performed on the item at idx=0.
        """
        # get the encoded item from the csv file at idx 0
        encoded_item = self.csv_loader[0]
        
        # test that the encoded item is a tuple of three dictionaries [input, label, meta]
        self.assertEqual(len(encoded_item), 3)
        self.assertIsInstance(encoded_item[0], dict)
        self.assertIsInstance(encoded_item[1], dict)
        self.assertIsInstance(encoded_item[2], dict)

        # check that the key of the encoded_item[0] (x) is the same as the key in the csv file
        self.assertEqual(list(encoded_item[0].keys())[0], "hello")

        # check that the key of the encoded_item[1] (y) is the same as the key in the csv file
        self.assertEqual(list(encoded_item[1].keys())[0], "hola")

        # check that the meta dictionary is empty 
        self.assertEqual(encoded_item[2], {})

        # since we retrieved only the data at idx=0, check that input and label both have only one element
        for key in encoded_item[0].keys():
            self.assertEqual(len(encoded_item[0][key]), 1)
        for key in encoded_item[1].keys():
            self.assertEqual(len(encoded_item[1][key]), 1)

    def test_get_encoded_item_multiple(self):
        """
        It tests that the csv_loader.get_encoded_item works well when getting multiple items using slice.
        The following test is performed on the item at idx=0 and idx=1.
        """
        
        # get the encoded items from the csv file at idx 0 and 1
        encoded_item = self.csv_loader[slice(0, 2)]
        
        # test that the encoded item is a tuple of three dictionaries [input, label, meta]
        self.assertEqual(len(encoded_item), 3)
        self.assertIsInstance(encoded_item[0], dict)
        self.assertIsInstance(encoded_item[1], dict)
        self.assertIsInstance(encoded_item[2], dict)

        # check that the key of the encoded_item[0] (x) is the same as the key in the csv file
        self.assertEqual(list(encoded_item[0].keys())[0], "hello")

        # check that the key of the encoded_item[1] (y) is the same as the key in the csv file
        self.assertEqual(list(encoded_item[1].keys())[0], "hola")

        # check that the meta dictionary is empty 
        self.assertEqual(encoded_item[2], {})
        
        # since we retrieved only the data at idx=0 and idx=1, check that input and label both have two elements
        for key in encoded_item[0].keys():
            self.assertEqual(len(encoded_item[0][key]), 2)
        for key in encoded_item[1].keys():
            self.assertEqual(len(encoded_item[1][key]), 2)

    def test_len(self):
        self.assertEqual(len(self.csv_loader), 2)

    def test_load_with_split(self):
        # test when split is not provided
        self.csv_loader_split = CsvLoader(ProtDnaToFloatExperiment(), os.path.abspath("bin/tests/test_data/test_with_split.csv"))
        self.assertEqual(len(self.csv_loader_split.input['hello:dna']), 3)

        # test when split is 0, 1 or 2
        for i in [0, 1, 2]:
            self.csv_loader_split = CsvLoader(ProtDnaToFloatExperiment(), os.path.abspath("bin/tests/test_data/test_with_split.csv"), split=i)
            self.assertEqual(len(self.csv_loader_split.input['hello:dna']), 1)
            self.assertEqual(self.csv_loader_split.split['split:int'][0], i)

        # test when split is not valid
        with self.assertRaises(ValueError): # should raise an error
            self.csv_loader_split = CsvLoader(ProtDnaToFloatExperiment(), os.path.abspath("bin/tests/test_data/test_with_split.csv"), split=3)
