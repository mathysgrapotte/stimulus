"""
unit test cases for the augmentation_generators file shold be written like the following

To write test cases for a new noise generator class:
1. Create a new test case class by subclassing unittest.TestCase.
2. Write test methods to test the behavior of the augmentation generator class methods.
3. Use assertions (e.g., self.assertIsInstance, self.assertEqual) to verify the behavior of the noise generator class methods.

Example:

class TestMyNoiseGenerator(unittest.TestCase):
    def test_add_augmentation(self):
        # Test adding noise to data
        augmentation_generator = MyAugmentationGenerator()
        augmented_data = augmentation_generator.add_augmentation(data)
        self.assertIsInstance(augmented_data, expected_type)
"""


import unittest
import numpy as np
from bin.src.data.augmentation.augmentation_generators import ReverseComplement


class TestReverseComplement(unittest.TestCase):
    def test_add_augmentation_single(self):
        # Test generation of augmentation to a single string
        augmenter = ReverseComplement()
        augmented_data = augmenter.add_augmentation("ACCCCTACGTNN")
        self.assertIsInstance(augmented_data, str)     
        self.assertEqual(augmented_data, "NNACGTAGGGGT")    

    def test_add_noise_multiprocess_with_single_item(self):
        # Test adding augmentation to a list of strings using multiprocessing, but when only one item is given
        augmenter = ReverseComplement()
        augmented_data_list = augmenter.add_augmentation_all(["ACCCCTACGTNN"] )
        self.assertIsInstance(augmented_data_list, list)    
        self.assertIsInstance(augmented_data_list[0], str)
        self.assertEqual(augmented_data_list, ['NNACGTAGGGGT']) 

    def test_add_noise_multiprocess_with_multiple_item(self):
        # Test adding noise to a list of strings using multiprocessing
        augmenter = ReverseComplement()
        augmented_data_list = augmenter.add_augmentation_all(["ACCCCTACGTNN", "ACTGA"] )
        self.assertIsInstance(augmented_data_list, list)     # making sure output is of correct type
        self.assertIsInstance(augmented_data_list[0], str)
        self.assertIsInstance(augmented_data_list[1], str)
        self.assertEqual(augmented_data_list, ['NNACGTAGGGGT', 'TCAGT'])    # checking if given a seed the noise happens in the same way

if __name__ == "__main__":
    unittest.main()