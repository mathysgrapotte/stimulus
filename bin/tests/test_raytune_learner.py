import unittest
import os
import torch
from bin.src.data.experiments import DnaToFloatExperiment
from bin.src.learner.raytune_learner import TuneModel as RayTuneLearner
from bin.tests.test_model.dnatofloatmodel import ModelSimple
from bin.src.learner.raytune_learner import TuneWrapper 
from torch.utils.data import DataLoader


class TestRayTuneLearner(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        config = {}
        with open("bin/tests/test_model/simple.config", "r") as f:
            config = eval(f.read())
        config["model"] = ModelSimple
        config["experiment"] = DnaToFloatExperiment()
        config["data_path"] = "bin/tests/test_data/dna_experiment/test_with_split.csv"
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
        os.remove("bin/tests/test_data/dna_experiment/test_checkpoint.pth")

class TestTuneWrapper(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        config_path = "bin/tests/test_model/simple.config"
        model_class = ModelSimple
        experiment_obj = DnaToFloatExperiment()
        data_path = "bin/tests/test_data/dna_experiment/test_with_split.csv"
        self.wrapper = TuneWrapper(config_path, model_class, data_path, experiment_obj)
    
    def test_setup(self):
        self.assertIsInstance(self.wrapper.config, dict)
        self.assertTrue(self.wrapper.tune_config is not None)
        self.assertTrue(self.wrapper.checkpoint_config is not None)
        self.assertTrue(self.wrapper.run_config is not None)
        self.assertTrue(self.wrapper.scheduler is not None)
    
    def test_prep_tuner(self):
        self.wrapper._prep_tuner()
        self.assertTrue(self.wrapper.tuner is not None)
    
    def test_tune(self):
        self.wrapper.tune()
        self.assertTrue(self.wrapper.results is not None)
    
    def test_store_best_config(self):
        self.wrapper.store_best_config("bin/tests/test_data/dna_experiment/best_config.json")
        self.assertTrue(os.path.exists("bin/tests/test_data/dna_experiment/best_config.json"))
        os.remove("bin/tests/test_data/dna_experiment/best_config.json")
    
    def test_train(self):
        self.wrapper.tune()
        self.wrapper.train()
        self.assertTrue(self.wrapper.trainer is not None)

if __name__ == "__main__":
    unittest.main()
    
    