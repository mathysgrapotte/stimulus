import numpy as np
import numpy.testing as npt
import unittest
from bin.src.data.csv_parser import CSVParser
from bin.src.data.experiments import DnaToFloatExperiment

class TestDnaToFloatCsvParser(unittest.TestCase):

    def setUp(self):
        self.csv_parser = CSVParser(DnaToFloatExperiment(), "test_data/test.csv")

    def test_get_encoded_item_unique(self):
        # Test getting an encoded item from the csv file
        encoded_item = self.csv_parser.get_encoded_item(0)
        self.assertIsInstance(encoded_item[0], dict)
        self.assertIsInstance(encoded_item[1], dict)
        self.assertIsInstance(encoded_item[2], dict)

        # check that the key of the encoded_item[0] (x) is the same as the key in the csv file
        self.assertEqual(list(encoded_item[0].keys())[0], "hello")

        # check that the key of the encoded_item[1] (y) is the same as the key in the csv file
        self.assertEqual(list(encoded_item[1].keys())[0], "hola")

        # check that the meta dictionary is empty 
        self.assertEqual(encoded_item[2], {})

        # check that x and y both have only one sequence
        self.assertEqual(len(list(encoded_item[0].values())[0]), 1)
        self.assertEqual(len(list(encoded_item[1].values())[0]), 1)



