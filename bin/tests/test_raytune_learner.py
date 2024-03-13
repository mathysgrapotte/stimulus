from bin.src.learner.raytune_learner import TuneModel as RayTuneLearner
from bin.tests.test_model.dnatofloatmodel import SimpleModel, CONFIG_EXAMPLE
from bin.src.data.experiments import DnaToFloatExperiment
import unittest
import os

class TestRayTuneLearner(unittest.TestCase):
    def setUp(self):
        config = CONFIG_EXAMPLE
        config["model"] = SimpleModel
        config["data_path"] = os.path.abspath("bin/tests/test_data/test_with_split.csv")
        config["experiment"] = DnaToFloatExperiment()
        self.learner = RayTuneLearner(config = config)

    def test_setup(self):
        self.assertTrue(self.learner.model is not None)
        self.assertTrue(self.learner.loss_fn is not None)
        self.assertTrue(self.learner.optimizer is not None)
        self.assertTrue(self.learner.train is not None)
        self.assertTrue(self.learner.test is not None)

    def test_step(self):
        self.learner.step()