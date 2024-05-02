"""
unit test cases for the noise_generators file shold be written like the following

Test case for the Splitter class.

To write test cases for a new noise generator class:
1. Create a new test case class by subclassing unittest.TestCase.
2. Write test methods to test the behavior of the noise generator class methods.
3. Use assertions (e.g., self.assertIsInstance, self.assertEqual) to verify the behavior of the noise generator class methods.

"""


import unittest
import numpy as np
from bin.src.data.splitters.splitters import RandomSplitter
import polars as pl

def sample_data():
    # Create a sample dataframe
    df = pl.DataFrame({
        'A': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'B': [6, 7, 8, 9, 10, 6, 7, 8, 9, 10]
    })
    return df

class TestRandomSpliter(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.sample_data = sample_data()

    
    def test_get_split_indexes(self):
        sample_data = self.sample_data
        splitter = RandomSplitter()
         # Test splitting with custom split proportions
        custom_split = [0.6, 0.3, 0.1]
        train_custom, validation_custom, test_custom = splitter.get_split_indexes(data=sample_data, split=custom_split, seed=123)
        # train is 0, count how many 
        self.assertEqual(len(train_custom), 6)
        self.assertEqual(len(validation_custom), 3)
        self.assertEqual(len(test_custom), 1)

if __name__ == "__main__":
    unittest.main()