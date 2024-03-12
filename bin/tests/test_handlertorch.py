import numpy as np
import numpy.testing as npt
import unittest
import os
import torch
from bin.src.data.handlertorch import TorchDataset
from bin.src.data.experiments import DnaToFloatExperiment

# initialize unittest class
class TestDnaToFloatTorchDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.torchdataset_same_length = TorchDataset(csvpath=os.path.abspath("bin/tests/test_data/test.csv"), experiment=DnaToFloatExperiment())
        self.torchdataset_different_length = TorchDataset(csvpath=os.path.abspath("bin/tests/test_data/test_unequal_dna_float.csv"), experiment=DnaToFloatExperiment())

    def test_len_same_length(self):
        self.assertEqual(len(self.torchdataset_same_length),2)

    def test_list_of_numpy_arrays_to_tensor_same_lengths(self):
        data = [np.array([1,2,3]), np.array([4,5,6])]
        tensor, mask = self.torchdataset_same_length.convert_list_of_numpy_arrays_to_tensor(data)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertIsNone(mask)

    def test_convert_dict_to_tensor_same_lengths(self):
        data = {"hello": [np.array([1,2,3]), np.array([4,5,6])], "hola": [np.array([1,2,3]), np.array([4,5,6])]}
        output_dict, mask_dict = self.torchdataset_same_length.convert_dict_to_tensor(data)
        self.assertIsInstance(output_dict["hello"], torch.Tensor)
        self.assertIsNone(mask_dict["hello"])
        self.assertIsInstance(output_dict["hola"], torch.Tensor)
        self.assertIsNone(mask_dict["hola"])


        input_data = self.torchdataset_same_length.parser.get_encoded_item(slice(0, 2))
        output_dict, mask_dict = self.torchdataset_same_length.convert_dict_to_tensor(input_data[0])

    def test_get_item_same_lenghts(self):
        x, y, x_mask, y_mask, meta = self.torchdataset_same_length[0:2]
        self.assertIsInstance(x, dict)
        self.assertIsInstance(y, dict)
        self.assertIsInstance(meta, dict)

        # now test for one item
        x, y, x_mask, y_mask, meta = self.torchdataset_same_length[0]
        self.assertIsInstance(x, dict)
        self.assertIsInstance(y, dict)
        self.assertIsInstance(meta, dict)

    def test_len_different_length(self):
        self.assertEqual(len(self.torchdataset_different_length),4)

    def test_list_of_numpy_arrays_to_tensor_different_lengths(self):
        data = [np.array([1,2,3]), np.array([4,5,6]), np.array([7,8,9,10])]
        tensor, mask = self.torchdataset_different_length.convert_list_of_numpy_arrays_to_tensor(data)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)

    def test_convert_dict_to_tensor_different_lengths(self):
        data = {"hello": [np.array([1,2,3]), np.array([4,5,6]), np.array([7,8,9,10])], "hola": [np.array([1,2,3]), np.array([4,5,6]), np.array([7,8,9,10])]}
        output_dict, mask_dict = self.torchdataset_different_length.convert_dict_to_tensor(data)
        self.assertIsInstance(output_dict["hello"], torch.Tensor)
        self.assertIsInstance(mask_dict["hello"], torch.Tensor)
        self.assertIsInstance(output_dict["hola"], torch.Tensor)
        self.assertIsInstance(mask_dict["hola"], torch.Tensor)

    def test_get_item_different_lenghts(self):
        x, y, x_mask, y_mask, meta = self.torchdataset_different_length[0:2]
        self.assertIsInstance(x, dict)
        self.assertIsInstance(y, dict)
        self.assertIsInstance(meta, dict)

        # now test for one item
        x, y, x_mask, y_mask, meta = self.torchdataset_different_length[0]
        self.assertIsInstance(x, dict)
        self.assertIsInstance(y, dict)
        self.assertIsInstance(meta, dict)


