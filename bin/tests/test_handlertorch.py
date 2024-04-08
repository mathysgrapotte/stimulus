import numpy as np
import unittest
import os
import torch
from bin.src.data.handlertorch import TorchDataset
from bin.src.data.experiments import DnaToFloatExperiment


class TestTorchDataset(unittest.TestCase):

    def _test_len(self, expected_len: int):
        self.assertEqual(len(self.torchdataset), expected_len)

    def _test_convert_dict_to_dict_of_tensor(self, data: dict, expected_len: dict):
        for key in data:
            self.assertIsInstance(data[key], torch.Tensor)
            self.assertEqual(data[key].shape, torch.Size(expected_len[key]))

    def _test_get_item(self, idx: int):
        x, y, meta = self.torchdataset[idx]
        self.assertIsInstance(x, dict)
        self.assertIsInstance(y, dict)
        self.assertIsInstance(meta, dict)

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
        self._test_get_item(0)

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
        self._test_get_item(0)



    # def test_get_item_same_lenghts(self):
    #     x, y, x_mask, y_mask, meta = self.torchdataset_same_length[0:2]
    #     self.assertIsInstance(x, dict)
    #     self.assertIsInstance(y, dict)
    #     self.assertIsInstance(meta, dict)

    #     # now test for one item
    #     x, y, x_mask, y_mask, meta = self.torchdataset_same_length[0]
    #     self.assertIsInstance(x, dict)
    #     self.assertIsInstance(y, dict)
    #     self.assertIsInstance(meta, dict)

    # def test_len_different_length(self):
    #     # test the length of the dataset with sequences of different e
    #     self.assertEqual(len(self.torchdataset_different_length),4)

    # def test_list_of_numpy_arrays_to_tensor_different_lengths(self):
    #     data = [np.array([1,2,3]), np.array([4,5,6]), np.array([7,8,9,10])]
    #     tensor, mask = self.torchdataset_different_length.convert_list_of_numpy_arrays_to_tensor(data)
    #     self.assertIsInstance(tensor, torch.Tensor)
    #     self.assertIsInstance(mask, torch.Tensor)



    # def test_convert_dict_to_tensor_different_lengths(self):
    #     data = {"hello": [np.array([1,2,3]), np.array([4,5,6]), np.array([7,8,9,10])], "hola": [np.array([1,2,3]), np.array([4,5,6]), np.array([7,8,9,10])]}
    #     output_dict, mask_dict = self.torchdataset_different_length.convert_dict_to_tensor(data)
    #     self.assertIsInstance(output_dict["hello"], torch.Tensor)
    #     self.assertIsInstance(mask_dict["hello"], torch.Tensor)
    #     self.assertIsInstance(output_dict["hola"], torch.Tensor)
    #     self.assertIsInstance(mask_dict["hola"], torch.Tensor)

    # def test_get_item_different_lenghts(self):
    #     x, y, x_mask, y_mask, meta = self.torchdataset_different_length[0:2]
    #     self.assertIsInstance(x, dict)
    #     self.assertIsInstance(y, dict)
    #     self.assertIsInstance(meta, dict)

    #     # now test for one item
    #     x, y, x_mask, y_mask, meta = self.torchdataset_different_length[0]
    #     self.assertIsInstance(x, dict)
    #     self.assertIsInstance(y, dict)
    #     self.assertIsInstance(meta, dict)
