import unittest
import os
from bin.src.data.csv import CsvLoader
from bin.src.data.experiments import DnaToFloatExperiment

class TestDnaToFloatCsvLoader(unittest.TestCase):

    def setUp(self):
        self.csv_loader = CsvLoader(DnaToFloatExperiment(), os.path.abspath("bin/tests/test_data/test.csv"))
        self.csv_loader_split = CsvLoader(DnaToFloatExperiment(), os.path.abspath("bin/tests/test_data/test_with_split.csv"), split=0)

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
        # try loading with different split values, should run with 0,1 and 2 and raise an error for other values
        self.csv_loader_split = CsvLoader(DnaToFloatExperiment(), os.path.abspath("bin/tests/test_data/test_with_split.csv"), split=0)
        # self.csv_loader_split.input['hello'] should have only one value 
        self.assertEqual(len(self.csv_loader_split.input['hello:dna']), 1)
        # check that self.csv_loader_split.meta has only one value in the ['split:int'] column which is 0
        self.assertEqual(len(self.csv_loader_split.meta['split:int']), 1)
        self.assertEqual(self.csv_loader_split.meta['split:int'][0], 0)
        self.csv_loader_split = CsvLoader(DnaToFloatExperiment(), os.path.abspath("bin/tests/test_data/test_with_split.csv"), split=1)
        self.csv_loader_split = CsvLoader(DnaToFloatExperiment(), os.path.abspath("bin/tests/test_data/test_with_split.csv"), split=2)
        with self.assertRaises(ValueError): # should raise an error
            self.csv_loader_split = CsvLoader(DnaToFloatExperiment(), os.path.abspath("bin/tests/test_data/test_with_split.csv"), split=3)

        
        