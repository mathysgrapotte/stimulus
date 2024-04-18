import unittest
import os
import torch
from bin.src.data.experiments import DnaToFloatExperiment
from bin.src.learner.raytune_learner import TuneModel as RayTuneLearner
from bin.tests.test_model.dnatofloatmodel import SimpleModel
from bin.src.learner.raytune_learner import TuneTrainWrapper 
from torch.utils.data import DataLoader
from ray import train, tune

class TestRayTuneLearner(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        config = {}
        with open("bin/tests/test_model/simple.config", "r") as f:
            config = eval(f.read())
        self.learner = RayTuneLearner(config = config)

    def test_setup(self):
        self.assertIsInstance(self.learner.loss_dict, dict)
        self.assertTrue(self.learner.optimizer is not None)
        self.assertIsInstance(self.learner.epochs, int)
        self.assertTrue(self.learner.lr is not None)
        self.assertIsInstance(self.learner.training, DataLoader)
        self.assertIsInstance(self.learner.validation, DataLoader) 

    def test_step(self):
        self.learner.step()
        test_data = next(iter(self.learner.training))[0]["hello"]
        test_output = self.learner.model(test_data)
        test_output = round(test_output.item(),4)
        self.assertEqual(test_output, 0.2298)

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
        os.remove("bin/tests/test_data/dna_experiment/test_model.pth")
    
    def test_save_checkpoint(self):
        checkpoint = self.learner.save_checkpoint("bin/tests/test_data/dna_experiment/test_checkpoint.pth")
        self.assertIsInstance(checkpoint, dict)
        self.assertTrue(os.path.exists("bin/tests/test_data/dna_experiment/test_checkpoint.pth"))
        os.remove("bin/tests/test_data/dna_experiment/test_checkpoint.pth")

    def test_load_checkpoint(self):
        self.learner.save_checkpoint("bin/tests/test_data/dna_experiment/test_checkpoint.pth")
        checkpoint = self.learner.save_checkpoint("bin/tests/test_data/dna_experiment/test_checkpoint.pth")
        self.learner.load_checkpoint(checkpoint)
        self.assertTrue(True)
        os.remove("bin/tests/test_data/dna_experiment/test_checkpoint.pth")


class TestTrainTuneWrapper(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        config_path = "bin/tests/test_model/simple.config"
        model_path = 'bin.tests.test_model.dnatofloatmodel.SimpleModel'
        experiment_path = 'bin.src.data.experiments.DnaToFloatExperiment'
        data_path = 'bin/tests/test_data/dna_experiment/test_with_split.csv'
        self.learner = TuneTrainWrapper(config_path, model_path, experiment_path, data_path)
    
    def test_setup(self):
        self.assertIsInstance(self.learner.config, dict)
        self.assertTrue(self.learner.tune_config is not None)
        self.assertTrue(self.learner.checkpoint_config is not None)
        self.assertTrue(self.learner.run_config is not None)
        self.assertTrue(self.learner.scheduler is not None)
        



if __name__ == "__main__":
    unittest.main()
    
    