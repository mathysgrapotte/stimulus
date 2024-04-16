import unittest
import os
from bin.src.data.experiments import DnaToFloatExperiment
from bin.src.learner.raytune_learner import TuneModel as RayTuneLearner
from bin.tests.test_model.dnatofloatmodel import SimpleModel, CONFIG_EXAMPLE
from torch.utils.data import DataLoader

class TestRayTuneLearner(unittest.TestCase):
    def setUp(self):
        config = CONFIG_EXAMPLE
        config["model"] = SimpleModel
        config["data_path"] = os.path.abspath("bin/tests/test_data/dna_experiment/test_with_split.csv")
        config["experiment"] = DnaToFloatExperiment()
        self.learner = RayTuneLearner(config = config)

    def test_setup(self):
        self.assertTrue(1==1)
        self.assertIsInstance(self.learner.loss_dict, dict)
        self.assertTrue(self.learner.optimizer is not None)
        self.assertIsInstance(self.learner.epochs, int)
        self.assertTrue(self.learner.lr is not None)
        self.assertIsInstance(self.learner.train, DataLoader)
        self.assertIsInstance(self.learner.validation, DataLoader) 

    # def test_step(self):
    #     self.learner.step()

    # def test_objective(self):
    #     self.learner.objective()

    # def test_compute_val_loss(self):
    #     self.learner.compute_val_loss()
    
    
        

if __name__ == "__main__":
    unittest.main()
    