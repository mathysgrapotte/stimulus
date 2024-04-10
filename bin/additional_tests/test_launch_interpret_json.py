import unittest
from bin.launch_interpret_json import interpret_json

""" 
to run this test you need to put a relatibve inmport in the JsonSchema import line in launch_interpret_json.py.

To explain this further launch_interpret_json.py is meant to be launched as it is:
python3 launch_interpret_json.py 

So it can not have relative imports inside, (it can but it gets complicated quickly). 
But here to test it we have to import it, and once we do that it will throw an error because the absolute import it has for
the JsonSchema class will not be resolved. 

Basically there is no simple way to have the louncher in bin keeping the opverall directory organization 
and a nice set of tests for it that live in another directory. Hence the need to manually mnodify that file when the need for test arise.

TODO find a clean solution at the above problem.
"""

# initialize unittest class
class TestInterpretJson(unittest.TestCase):

    def test_interpret_json_with_empty_json(self):
        d = {"experiment": "MyCustomExperiment"}
        out_l = [{"experiment": "MyCustomExperiment", "noise": None, "split": None}]
        self.assertEqual(interpret_json(d), out_l)
    

    def test_interpret_json_without_noise_arg(self):
        d = {
            "experiment": "MyCustomExperiment",
            "split": [
                {
                    "name": "RandomSplitter",
                    "params": [{"split": [[0.6, 0.2, 0.2], [0.7, 0.15, 0.15]]}]
                },
                {
                    "name": "SomeSplitter",
                    "params": "default"
                },
                {
                    "name": "SomeSplitter1",
                    "params": ["default"]
            }]}
        out_l = [{'experiment': 'MyCustomExperiment', 'noise': None, 'split': None},
        {'experiment': 'MyCustomExperiment', 'noise': None, 'split': {'name': 'RandomSplitter', 'params': {'split': [0.6, 0.2, 0.2]}}},
        {'experiment': 'MyCustomExperiment', 'noise': None, 'split': {'name': 'RandomSplitter', 'params': {'split': [0.7, 0.15, 0.15]}}},
        {'experiment': 'MyCustomExperiment', 'noise': None, 'split': {'name': 'SomeSplitter', 'params': {}}},
        {'experiment': 'MyCustomExperiment', 'noise': None, 'split': {'name': 'SomeSplitter1', 'params': {}}}]
        self.assertEqual(interpret_json(d), out_l)

    
    def test_interpret_json_without_split_arg_column_wise(self):
        d = {
            "experiment": "MyCustomExperiment",
            "interpret_params_mode": "column_wise", 
            "noise": [
                {
                    "column_name": "hello:input1:dna",
                    "name": ["UniformTextMasker", "AnotherNoiser", "AnotherNoiser"],
                    "params": [{"probability": [0.1, 0.2]}, "default", {"probability": [0.12, 0.22], "seed": [0, 0]}]
                },
                {
                    "column_name": "hello:input2:prot",
                    "name": ["UniformTextMasker", "AnotherNoiser1"],
                    "params": ["default", {"p": [1, 2], "s": [3, 4]}]
                },
                {
                    "column_name": "hello:label:int",
                    "name": ["YetAnotherNoiser"],
                    "params": "default"
                }]}
        out_l =[{'experiment': 'MyCustomExperiment', 'noise': None, 'split': None},
        {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'UniformTextMasker', 'params': {'probability': 0.1}}, {'column_name': 'hello:input2:prot', 'name': 'UniformTextMasker', 'params': {}}, {'column_name': 'hello:label:int', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': None},
        {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'UniformTextMasker', 'params': {'probability': 0.2}}, {'column_name': 'hello:input2:prot', 'name': 'UniformTextMasker', 'params': {}}, {'column_name': 'hello:label:int', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': None}, 
        {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'UniformTextMasker', 'params': {'probability': 0.1}}, {'column_name': 'hello:input2:prot', 'name': 'AnotherNoiser1', 'params': {'p': 1, 's': 3}}, {'column_name': 'hello:label:int', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': None}, 
        {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'UniformTextMasker', 'params': {'probability': 0.2}}, {'column_name': 'hello:input2:prot', 'name': 'AnotherNoiser1', 'params': {'p': 2, 's': 4}}, {'column_name': 'hello:label:int', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': None}, 
        {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'AnotherNoiser', 'params': {}}, {'column_name': 'hello:input2:prot', 'name': 'UniformTextMasker', 'params': {}}, {'column_name': 'hello:label:int', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': None}, 
        {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'AnotherNoiser', 'params': {}}, {'column_name': 'hello:input2:prot', 'name': 'AnotherNoiser1', 'params': {'p': 1, 's': 3}}, {'column_name': 'hello:label:int', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': None}, 
        {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'AnotherNoiser', 'params': {}}, {'column_name': 'hello:input2:prot', 'name': 'AnotherNoiser1', 'params': {'p': 2, 's': 4}}, {'column_name': 'hello:label:int', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': None}, 
        {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'AnotherNoiser', 'params': {'probability': 0.12, 'seed': 0}}, {'column_name': 'hello:input2:prot', 'name': 'UniformTextMasker', 'params': {}}, {'column_name': 'hello:label:int', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': None}, 
        {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'AnotherNoiser', 'params': {'probability': 0.22, 'seed': 0}}, {'column_name': 'hello:input2:prot', 'name': 'UniformTextMasker', 'params': {}}, {'column_name': 'hello:label:int', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': None}, 
        {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'AnotherNoiser', 'params': {'probability': 0.12, 'seed': 0}}, {'column_name': 'hello:input2:prot', 'name': 'AnotherNoiser1', 'params': {'p': 1, 's': 3}}, {'column_name': 'hello:label:int', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': None}, 
        {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'AnotherNoiser', 'params': {'probability': 0.22, 'seed': 0}}, {'column_name': 'hello:input2:prot', 'name': 'AnotherNoiser1', 'params': {'p': 2, 's': 4}}, {'column_name': 'hello:label:int', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': None}]
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

    
    def test_interpret_json_with_column_wise(self):
        d = {
            "experiment": "MyCustomExperiment",
            "interpret_params_mode": "column_wise", 
            "noise": [
                {
                    "column_name": "hello:input1:dna",
                    "name": ["UniformTextMasker", "AnotherNoiser"],
                    "params": ["default", {"mean": [0.5, 0.6], "std": [0.1, 0.2]}]
                },
                {
                    "column_name": "hello:input2:prot",
                    "name": ["YetAnotherNoiser"],
                    "params": ["default", {"p1": [1, 2]}]
                }],
            "split": [
                {
                    "name": "RandomSplitter",
                    "params": [{"split": [[0.6, 0.2, 0.2], [0.7, 0.15, 0.15]]}]
                },
                {
                    "name": "SomeSplitter",
                    "params": "default"
            }]}

        
        out_list = [
            {'experiment': 'MyCustomExperiment', 'noise': None, 'split': None},
            {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'UniformTextMasker', 'params': {}}, {'column_name': 'hello:input2:prot', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': {'name': 'RandomSplitter', 'params': {'split': [0.6, 0.2, 0.2]}}},
            {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'UniformTextMasker', 'params': {}}, {'column_name': 'hello:input2:prot', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': {'name': 'RandomSplitter', 'params': {'split': [0.7, 0.15, 0.15]}}},
            {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'UniformTextMasker', 'params': {}}, {'column_name': 'hello:input2:prot', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': {'name': 'SomeSplitter', 'params': {}}},
            {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'AnotherNoiser', 'params': {'mean': 0.5, 'std': 0.1}}, {'column_name': 'hello:input2:prot', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': {'name': 'RandomSplitter', 'params': {'split': [0.6, 0.2, 0.2]}}},
            {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'AnotherNoiser', 'params': {'mean': 0.5, 'std': 0.1}}, {'column_name': 'hello:input2:prot', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': {'name': 'RandomSplitter', 'params': {'split': [0.7, 0.15, 0.15]}}},
            {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'AnotherNoiser', 'params': {'mean': 0.5, 'std': 0.1}}, {'column_name': 'hello:input2:prot', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': {'name': 'SomeSplitter', 'params': {}}},
            {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'AnotherNoiser', 'params': {'mean': 0.6, 'std': 0.2}}, {'column_name': 'hello:input2:prot', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': {'name': 'RandomSplitter', 'params': {'split': [0.6, 0.2, 0.2]}}},
            {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'AnotherNoiser', 'params': {'mean': 0.6, 'std': 0.2}}, {'column_name': 'hello:input2:prot', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': {'name': 'RandomSplitter', 'params': {'split': [0.7, 0.15, 0.15]}}},
            {'experiment': 'MyCustomExperiment', 'noise': [{'column_name': 'hello:input1:dna', 'name': 'AnotherNoiser', 'params': {'mean': 0.6, 'std': 0.2}}, {'column_name': 'hello:input2:prot', 'name': 'YetAnotherNoiser', 'params': {}}], 'split': {'name': 'SomeSplitter', 'params': {}}}
            ]

        d_to_test = interpret_json(d)
        self.assertEqual(len(d_to_test), 10)
        self.assertEqual(d_to_test, out_list)
