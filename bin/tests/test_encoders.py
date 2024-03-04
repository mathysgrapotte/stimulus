"""
unit test cases for the encoders file

Test case for the TextOneHotEncoder class.

To create a similar test case for another class with a similar structure:
1. Create a new test case class by subclassing unittest.TestCase.
2. Add setup code in the setUp method to create an instance of the class being tested.
3. Write test methods to test the functionality of the class methods.
4. Use assertions (e.g., self.assertEqual, self.assertIsInstance) to verify the behavior of the class methods.
5. update the bin/requirements.txt file with the new dependencies neede from the class, if any new is added.

Example:

class TestMyClass(unittest.TestCase):
    def setUp(self):
        # Create an instance of the class being tested
        self.my_class_instance = MyClass()

    def test_method1(self):
        # Test method 1 of the class
        result = self.my_class_instance.method1()
        self.assertEqual(result, expected_result)

    def test_method2(self):
        # Test method 2 of the class
        result = self.my_class_instance.method2()
        self.assertIsInstance(result, expected_type)
"""

import numpy as np
import unittest
from src.data.types.encoding.encoders import TextOneHotEncoder


class TestTextOneHotEncoder(unittest.TestCase):

    def setUp(self):
        self.text_encoder = TextOneHotEncoder()

    def test_encode(self):
        # Test encoding a valid sequence
        encoded_data = self.text_encoder.encode("ACGT")
        self.assertIsInstance(encoded_data, np.ndarray)
        self.assertEqual(encoded_data.shape, (4, 4))  # Expected shape for one-hot encoding of "ACGT"

        # Test encoding an empty sequence
        encoded_data_empty = self.text_encoder.encode("")
        self.assertIsInstance(encoded_data_empty, np.ndarray)
        self.assertEqual(encoded_data_empty.shape, (0, 4))  # Expected shape for one-hot encoding of an empty sequence

    def test_decode(self):
        # Test decoding a one-hot encoded sequence
        encoded_data = self.text_encoder.encode("ACGT")
        decoded_sequence = self.text_encoder.decode(encoded_data)
        self.assertIsInstance(decoded_sequence, np.ndarray)
        self.assertEqual(decoded_sequence.shape, (4, 1))  # Expected shape for the decoded sequence
        self.assertEqual("".join(decoded_sequence.flatten()), "acgt")  # Expected decoded sequence

        # Test decoding an empty one-hot encoded sequence
        encoded_data_empty = np.array([])
        decoded_sequence_empty = self.text_encoder.decode(encoded_data_empty)
        self.assertIsInstance(decoded_sequence_empty, np.ndarray)
        self.assertEqual(decoded_sequence_empty.size, 0)  # Expected size for the decoded sequence

if __name__ == "__main__":
    unittest.main()
