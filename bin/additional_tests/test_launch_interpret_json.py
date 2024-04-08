import unittest
from bin.launch_interpret_json import interpret_json

""" 
to run this test you need to put a relatibve inmport in the JsonSchema import line in launch_interpret_json
"""

# initialize unittest class
class TestInterpretJson(unittest.TestCase):

    def test_interpret_json_with_empty_json(self):
        d = {"experiment": "MyCustomExperiment"}
        out_l = [{"experiment": "MyCustomExperiment", "noise": None, "split": None}]
        self.assertEqual(interpret_json(d), out_l)

    def test_interpret_json_with_custom_dict(self):
        d = {
            "experiment": "MyCustomExperiment",
            "custom": [
                {
                    "noise": [
                        {
                            "column_name": "input1",
                            "name": "UniformTextMasker",
                            "params": {"probability": 0.1}
                        },
                        {
                            "column_name": "input2",
                            "name": "GaussianNoise",
                            "params": {"mean": 0.5, "std": 0.1}
                        }],
                    "split": [
                        {
                            "name": "RandomSplitter",
                            "params": {"split": [0.6, 0.4, 0]}
                        }]},
                {
                    "noise": [
                        {
                            "column_name": "input2",
                            "name": "UniformTextMasker",
                            "params": {"probability": 0.1}
                        },
                        {
                            "column_name": "float",
                            "name": "GaussianNoise",
                            "params": {"mean": 0.5, "std": 0.1}
                        }],
                    "split": [
                        {
                            "name": "RandomSplitter",
                            "params": {"split": [0.6, 0.8, 0.1]}
            }]}]}
        
        out_l = [{'experiment': 'MyCustomExperiment', 'noise': None, 'split': None},
        {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'input1', 'name': 'UniformTextMasker', 'params': {'probability': 0.1}}, {'column_name': 'input2', 'name': 'GaussianNoise', 'params': {'mean': 0.5, 'std': 0.1}}], 'split': [{'name': 'RandomSplitter', 'params': {'split': [0.6, 0.4, 0]}}]},
        {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'input2', 'name': 'UniformTextMasker', 'params': {'probability': 0.1}}, {'column_name': 'float', 'name': 'GaussianNoise', 'params': {'mean': 0.5, 'std': 0.1}}], 'split': [{'name': 'RandomSplitter', 'params': {'split': [0.6, 0.8, 0.1]}}]}]
        self.assertEqual(interpret_json(d), out_l)