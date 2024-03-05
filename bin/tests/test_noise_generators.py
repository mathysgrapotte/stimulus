"""
unit test cases for the noise_generators file shold be written like the following

Test case for the GaussianNoise class.

To write test cases for a new noise generator class:
1. Create a new test case class by subclassing unittest.TestCase.
2. Write test methods to test the behavior of the noise generator class methods.
3. Use assertions (e.g., self.assertIsInstance, self.assertEqual) to verify the behavior of the noise generator class methods.

Example:

class TestMyNoiseGenerator(unittest.TestCase):
    def test_add_noise(self):
        # Test adding noise to data
        noise_generator = MyNoiseGenerator()
        noisy_data = noise_generator.add_noise(data)
        self.assertIsInstance(noisy_data, expected_type)
"""


import unittest
import numpy.testing as npt
from bin.src.data.data_types.noise.noise_generators import UniformTextMasker, GaussianNoise

class TestUniformTextMasker(unittest.TestCase):
    def test_add_noise_single(self):
        # Test adding noise to a single string
        masker = UniformTextMasker(probability=0.1)
        noisy_data = masker.add_noise("ACGTACGT", seed=42)
        self.assertIsInstance(noisy_data, str)      # making sure output is of correct type
        self.assertEqual(noisy_data, "ACGTACNT")    # checking if given a seed the noise happens in the same way

    def test_add_noise_multiprocess(self):
        # Test adding noise to a list of strings using multiprocessing
        masker = UniformTextMasker(probability=0.1)
        noisy_data_list = masker.add_noise_multiprocess(["ATCGATCGATCG", "ATCG"], seed=42)
        print(noisy_data_list)
        self.assertIsInstance(noisy_data_list, list)     # making sure output is of correct type
        self.assertIsInstance(noisy_data_list[0], str)
        self.assertIsInstance(noisy_data_list[1], str)
        self.assertEqual(noisy_data_list, ['ATCGATNGATNG', 'ATCG'])    # checking if given a seed the noise happens in the same way


class TestGaussianNoise(unittest.TestCase):
    def test_add_noise_single(self):
        # Test adding noise to a single float value
        noise_generator = GaussianNoise(mean=0, std=1)
        noisy_data = noise_generator.add_noise(5.0, seed=42)
        self.assertIsInstance(noisy_data, float)
        self.assertAlmostEqual(noisy_data, 5.4967141530)     # there might be float point variation across systems so not all decimals have to be identical

    def test_add_noise_multiprocess(self):
        # Test adding noise to a list of float values using multiprocessing
        noise_generator = GaussianNoise(mean=0, std=1)
        noisy_data_list = noise_generator.add_noise_multiprocess([1.0, 2.0, 3.0], seed=42)
        self.assertIsInstance(noisy_data_list, list)
        self.assertIsInstance(noisy_data_list[0], float)
        self.assertIsInstance(noisy_data_list[1], float)
        self.assertIsInstance(noisy_data_list[2], float)
        # using numpy testing because it looks at differences between arrays better
        npt.assert_almost_equal(noisy_data_list, [1.4967142, 1.8617357, 3.6476885], decimal=7, err_msg='The values  in the output array are not close to what is expected')  

        

if __name__ == "__main__":
    unittest.main()