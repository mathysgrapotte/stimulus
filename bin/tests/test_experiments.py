import numpy as np
import numpy.testing as npt
import unittest
from bin.src.data.experiments import DnaToFloatExperiment
from copy import deepcopy

class TestDnaToFloatExperiment(unittest.TestCase):
    
        def setUp(self):
            self.dna_to_float_experiment = DnaToFloatExperiment()

        def test_noise(self):
            # Test calling the noise method using a kwargs dictionary
            noise_method_noise_dna_uniform_masker = 'noise_dna_uniform_masker'
            kwarg_dict = {'probability': 0.5}
            original_data = {
                "sequences1:dna:input": ["ACGTACGT", "ACGTACGT", "ACGTACGT", "ACGTACGT"],
                "sequences2:dna:input": ["ACGTACGT", "ACGTACGT", "ACGTACGT", "ACGTACGT"],
                "float1:float:label": [1.0, 2.0, 3.0, 4.0],
                "float2:float:label": [1.0, 2.0, 3.0, 4.0]
            }

            data = deepcopy(original_data)
            noisy_data = self.dna_to_float_experiment.noise(data, noise_method_noise_dna_uniform_masker, **kwarg_dict)
            self.assertIsInstance(noisy_data, dict)
            self.assertEqual(len(noisy_data), 4)
            # checking if the noise was applied to the correct keys, meaning that the sequences have changed
            self.assertNotEqual(noisy_data["sequences1:dna:input"], original_data["sequences1:dna:input"])
            self.assertNotEqual(noisy_data["sequences2:dna:input"], original_data["sequences2:dna:input"])
            # checking if the noise was not applied to the float keys
            self.assertEqual(noisy_data["float1:float:label"], original_data["float1:float:label"])
            self.assertEqual(noisy_data["float2:float:label"], original_data["float2:float:label"])

