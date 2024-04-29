import numpy as np
import unittest
import os
import torch
from typing import Any, Tuple, Union, Literal
from bin.src.data.handlertorch import TorchDataset
from bin.src.data.experiments import DnaToFloatExperiment, ProtDnaToFloatExperiment, TitanicExperiment

class TestTorchDataset(unittest.TestCase):

    def _test_len(self, expected_len: int):
        self.assertEqual(len(self.torchdataset), expected_len)

    def _test_convert_dict_to_dict_of_tensor(self, data: dict, expected_len: dict):
        for key in data:
            self.assertIsInstance(data[key], torch.Tensor)
            self.assertEqual(data[key].shape, torch.Size(expected_len[key]))

    def _test_get_item_shape(self, idx: Any, expected_size: dict):
        x, y, meta = self.torchdataset[idx]
        self.assertIsInstance(x, dict)
        self.assertIsInstance(y, dict)
        self.assertIsInstance(meta, dict)
        for key,value in {**x, **y, **meta}.items():
            if key in expected_size:
                self.assertEqual(value.shape, torch.Size(expected_size[key]))

class TestDnaToFloatTorchDatasetSameLength(TestTorchDataset):

    def setUp(self) -> None:
        self.torchdataset = TorchDataset(csvpath=os.path.abspath("bin/tests/test_data/dna_experiment/test.csv"), experiment=DnaToFloatExperiment())

    def test_len(self):
        self._test_len(2)
    
    def test_convert_dict_to_dict_of_tensor(self):
        self._test_convert_dict_to_dict_of_tensor(
            self.torchdataset.input,
            {"hello": [2, 16, 4]}
        )
        self._test_convert_dict_to_dict_of_tensor(
            self.torchdataset.label,
            {"hola": [2]}
        )
    
    def test_get_item(self):
        self._test_get_item_shape(0, expected_size = {'hello': [16, 4]})  # 'hola': tensor(12.) has no shape
        self._test_get_item_shape(slice(0,2), expected_size={'hello': [2, 16, 4], 'hola': [2]})

class TestDnaToFloatTorchDatasetDifferentLength(TestTorchDataset):

    def setUp(self) -> None:
        self.torchdataset = TorchDataset(csvpath=os.path.abspath("bin/tests/test_data/dna_experiment/test_unequal_dna_float.csv"), experiment=DnaToFloatExperiment())

    def test_len(self):
        self._test_len(4)

    def test_convert_dict_to_dict_of_tensor(self):
        self._test_convert_dict_to_dict_of_tensor(
            self.torchdataset.input,
            {"hello": [4, 31, 4]}
        )
        self._test_convert_dict_to_dict_of_tensor(
            self.torchdataset.label,
            {"hola": [4]}
        )

    def test_get_item(self):
        self._test_get_item_shape(0, expected_size={'hello': [31, 4]})
        self._test_get_item_shape(slice(0,2), expected_size={'hello': [2, 31, 4], 'hola': [2]})

class TestProtDnaToFloatTorchDatasetSameLength(TestTorchDataset):

    def setUp(self) -> None:
        self.torchdataset = TorchDataset(csvpath=os.path.abspath("bin/tests/test_data/prot_dna_experiment/test.csv"), experiment=ProtDnaToFloatExperiment())

    def test_len(self):
        self._test_len(2)
    
    def test_convert_dict_to_dict_of_tensor(self):
        self._test_convert_dict_to_dict_of_tensor(
            self.torchdataset.input,
            {"hello": [2, 16, 4], "bonjour": [2, 15, 20]}
        )
        self._test_convert_dict_to_dict_of_tensor(
            self.torchdataset.label,
            {"hola": [2]}
        )
    
    def test_get_item(self):
        self._test_get_item_shape(0, expected_size = {'hello': [16, 4], 'bonjour': [15, 20]})
        self._test_get_item_shape(slice(0,2), expected_size={'hello': [2, 16, 4], 'bonjour': [2, 15, 20], 'hola': [2]})

class TestTitanicTorchDataset(TestTorchDataset):

    def setUp(self) -> None:
        self.torchdataset = TorchDataset(csvpath=os.path.abspath("bin/tests/test_data/titanic/titanic_stimulus.csv"), experiment=TitanicExperiment())

    def test_len(self):
        self._test_len(712)

if __name__ == "__main__":
    unittest.main()