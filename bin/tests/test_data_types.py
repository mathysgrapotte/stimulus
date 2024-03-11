import numpy as np
import numpy.testing as npt
import unittest
from bin.src.data.data_types.data_types import Dna, Prot

class TestDna(unittest.TestCase):

    def setUp(self):
        self.dna = Dna()

    def test_encode_all(self):
        """
        Test if the encode_all method runs with default arguments
        """
        # encode a list of sequences
        encoded_data_list = self.dna.encode_all(["ACGT", "AAA", "tt", "Bubba"])

        # check that the encoding returns a list
        self.assertIsInstance(encoded_data_list, list)

        # check if the arrays have the correct shape
        self.assertEqual(encoded_data_list[0].shape, (4, 4))

        # check we get the correct encoded arrays - first sequence
        correct_output = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        npt.assert_array_equal(encoded_data_list[0], correct_output)
        
    def test_encode_all_error(self):
        """
        Test if the encode_all method returns an error when the specified encoder is not within the list of possible encoders
        """
        with self.assertRaises(ValueError):
            self.dna.encode_all(["ACGT", "AAA", "tt", "Bubba"], encoder="not_a_valid_encoder")


class TestProt(unittest.TestCase):

    def setUp(self):
        self.prot = Prot()

    def test_encode_all(self):
        """
        Test if the encode_all method runs with default arguments
        acdefghiklmnpqrstvwy
        """
        # encode a list of sequences
        encoded_data_list = self.prot.encode_all(["ACDE", "FFF", "gg", "uuu"])

        # check that the encoding returns a list
        self.assertIsInstance(encoded_data_list, list)

        # check if the arrays have the correct shape
        self.assertEqual(encoded_data_list[0].shape, (4, 20))

        # check we get the correct encoded array - first sequence
        correct_output = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        npt.assert_array_equal(encoded_data_list[0], correct_output)
        
    def test_encode_all_error(self):
        """
        Test if the encode_all method returns an error when the specified encoder is not within the list of possible encoders
        """
        with self.assertRaises(ValueError):
            self.prot.encode_all(["ACDE", "FFF", "gg", "uuu"], encoder="not_a_valid_encoder")