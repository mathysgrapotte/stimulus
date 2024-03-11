import numpy as np
import numpy.testing as npt
import unittest
from bin.src.data.experiments import DnaToFloatExperiment
from copy import deepcopy

class TestDnaToFloatExperiment(unittest.TestCase):
    
        def setUp(self):
            self.dna_to_float_experiment = DnaToFloatExperiment()

        

