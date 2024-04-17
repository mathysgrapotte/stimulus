import unittest
import os
import random
import numpy as np
from bin.src.data.experiments import DnaToFloatExperiment
from bin.src.learner.raytune_learner import TuneModel as RayTuneLearner
from bin.tests.test_model.dnatofloatmodel import SimpleModel
from torch.utils.data import DataLoader

class TestRayTuneLearner(unittest.TestCase):
    def setUp(self):
        random.seed(1234)
        np.random.seed(1234)
        config = {}
        with open("bin/tests/test_model/simple.config", "r") as f:
            config = eval(f.read())
        self.learner = RayTuneLearner(config = config)

    def test_setup(self):
        self.assertIsInstance(self.learner.loss_dict, dict)
        self.assertTrue(self.learner.optimizer is not None)
        self.assertIsInstance(self.learner.epochs, int)
        self.assertTrue(self.learner.lr is not None)
        self.assertIsInstance(self.learner.train, DataLoader)
        self.assertIsInstance(self.learner.validation, DataLoader) 

    def test_step(self):
        self.learner.step()

    def test_objective(self):
        obj = self.learner.objective()
        self.assertIsInstance(obj, dict)
        self.assertTrue("val_loss" in obj.keys())
        self.assertIsInstance(obj["val_loss"], float)

    def test_compute_val_loss(self):
        val_loss = self.learner.compute_val_loss()
        self.assertIsInstance(val_loss, float)

    def test_export_model(self):
        self.learner.export_model("bin/tests/test_data/dna_experiment/test_model.pth")
        self.assertTrue(os.path.exists("bin/tests/test_data/dna_experiment/test_model.pth"))
    
    def test_save_checkpoint(self):
        checkpoint = self.learner.save_checkpoint("bin/tests/test_data/dna_experiment/test_checkpoint.pth")
        self.assertIsInstance(checkpoint, dict)
        self.assertTrue(os.path.exists("bin/tests/test_data/dna_experiment/test_checkpoint.pth"))

    def test_load_checkpoint(self):
        checkpoint = self.learner.save_checkpoint("bin/tests/test_data/dna_experiment/test_checkpoint.pth")
        self.learner.load_checkpoint(checkpoint)
        self.assertTrue(True)
    
    def tearDown(self):
        os.remove("bin/tests/test_data/dna_experiment/test_model.pth")
        os.remove("bin/tests/test_data/dna_experiment/test_checkpoint.pth")
    
    


if __name__ == "__main__":
    unittest.main()
    