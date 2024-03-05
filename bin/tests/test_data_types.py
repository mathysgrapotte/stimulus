import numpy as np
import numpy.testing as npt
import unittest
from bin.src.data.data_types.data_types import Dna

class TestDna(unittest.TestCase):

    def setUp(self):
        self.dna = Dna()

    # test if the encode_all method runs with default arguments
    def test_encode_all(self):
        # Test encoding a valid list of sequences
        encoded_data_list = self.dna.encode_all(["ACGT", "AAA", "tt", "Bubba"])
        self.assertIsInstance(encoded_data_list, list)
        # check if the length of the list is 4
        self.assertEqual(len(encoded_data_list), 4)
        correct_output = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        npt.assert_array_equal(encoded_data_list[0], correct_output)
        
    # test if the encode_all method returns an error when the specified encoder is not within the list of possible encoders
    def test_encode_all_error(self):
        # Test encoding a valid list of sequences
        with self.assertRaises(ValueError):
            self.dna.encode_all(["ACGT", "AAA", "tt", "Bubba"], encoder="not_a_valid_encoder")