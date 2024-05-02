import argparse
import torch.nn as nn
import torch
import os
import src.data.handlertorch as handlertorch
import yaml
import ray.tune as tune

from src.data.experiments import TitanicExperiment
from copy import deepcopy
from collections.abc import Callable



class YamlRayConfigLoader():
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            self.config = self.get_config(self.config)



    def raytune_space_selector(self, mode: Callable, space: list) -> Callable:
        # this function applies the mode function to the space, it needs to convert the space in a right way based on the mode, for instance, if the mode is "randint", the space should be a tuple of two integers and passed as *args

        if mode.__name__ == "choice":
            return mode(space)
        
        elif mode.__name__ in ["uniform", "loguniform", "quniform", "qloguniform", "qnormal", "randint"]:
            return mode(*tuple(space))
        
        else:
            raise NotImplementedError(f"Mode {mode.__name__} not implemented yet")



    def convert_raytune(self, param: dict) -> dict:
        # get the mode function from ray.tune using getattr, return an error if it is not recognized
        try:
            mode = getattr(tune, param["mode"])
        except AttributeError:
            raise AttributeError(f"Mode {param['mode']} not recognized, check the ray.tune documentation at https://docs.ray.io/en/master/tune/api_docs/suggestion.html")

        # apply the mode function to the space
        return self.raytune_space_selector(mode, param["space"])
    
    def get_config(self, config: dict) -> dict:
        # the config is a dictionary of dictionaries. The main dictionary keys are either model_params, loss_params or optimizer_params. 
        # The sub-dictionary keys are the parameters of the model, loss or optimizer, those params include two values, space and mode.
        # The space is the range of values to be tested, and the mode is the type of search to be done.
        # We convert the Yaml config by calling the correct function from ray.tune matching the mode, applied on the space
        # We return the config as a dictionary of dictionaries, where the values are the converted values from the space.

        new_config = deepcopy(config)
        for key in self.config:
            for sub_key in config[key]:
                new_config[key][sub_key] = self.convert_raytune(config[key][sub_key])

        return new_config
    
    def get_config_instance(self) -> dict:
        # this function take a config as input and returns an instance of said config with the values sampled from the space
        # the config is a dictionary of dictionaries. The main dictionary keys are either model_params, loss_params or optimizer_params.
        # The sub-dictionary keys are the parameters of the model, loss or optimizer, those params include two values, space and mode.

        config_instance = deepcopy(self.config)
        for key in self.config:
            for sub_key in self.config[key]:
                config_instance[key][sub_key] = self.config[key][sub_key].sample()

        return config_instance 


class ModelTitanic(nn.Module):
    def __init__(self, nb_neurons_intermediate_layer: int = 7, nb_intermediate_layers: int = 3, nb_classes: int = 2):
        super(ModelTitanic, self).__init__()
        self.input_layer = nn.Linear(7, nb_neurons_intermediate_layer)
        self.intermediate = nn.modules.ModuleList([nn.Linear(nb_neurons_intermediate_layer, nb_neurons_intermediate_layer) for _ in range(nb_intermediate_layers)])
        self.output_layer = nn.Linear(nb_neurons_intermediate_layer, nb_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pclass, sex, age, sibsp, parch, fare, embarked):
        # print all inputs
        x = torch.stack((pclass, sex, age, sibsp, parch, fare, embarked), dim=1).float()
        x = self.relu(self.input_layer(x))
        for layer in self.intermediate:
            x = self.relu(layer(x))
        x = self.softmax(self.output_layer(x))
        return x
    
    def compute_loss(self, loss_fn, output, survived):
        return loss_fn(output, survived)
    
    def epoch(self, x, y, loss_fn, optimizer):
        # concatenate all input tensors into one tensor that should be shape (batch_size, 7)

        output = self.forward(**x)
        loss = self.compute_loss(loss_fn, output, **y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
    
class CheckModelWrapper:
    def __init__(self, model: nn.Module, config_instance: dict, data_path: str):
        self.model = model(**config_instance["model_params"])
        # get the optimizer from pytorch
        optimizer = getattr(torch.optim, config_instance["optimizer"]["method"])
        # get the loss function from pytorch
        self.loss_fn = getattr(torch.nn, config_instance["loss_params"]["loss_fn"])()
        # instantiate the optimizer
        self.optimizer = optimizer(self.model.parameters(), **config_instance["optimizer_params"])
        # train_data is a TorchDataset object
        self.train_data = torch.utils.data.DataLoader(handlertorch.TorchDataset(os.path.abspath(data_path), TitanicExperiment(), split=None), batch_size=2, shuffle=True)

    def check_model(self):
        # get the initial model weights into a variable 
        initial_model_weights = deepcopy(self.model.state_dict())
        # load one sample of the data

        x, y, meta = next(iter(self.train_data))
        # train the model for one epoch
        loss = self.model.epoch(x, y, self.loss_fn, self.optimizer)
        # check the model weights have changed, and print if it has
        for key in initial_model_weights:
            if torch.equal(initial_model_weights[key], self.model.state_dict()[key]):
                print(f"Model weights have not changed for key {key}")
            else:
                print(f"Model weights have changed for key {key}")

        # print the computed loss, displaying loss_fn name as well
        print(f"Loss computed with {self.loss_fn.__class__.__name__} is {loss.item()}")


def arg_parser():
    parser = argparse.ArgumentParser(description="Launch check_model")
    parser.add_argument("--input", type=str, help="Path to input csv file", required=True)
    parser.add_argument("--model", type=str, help="Path to model file", required=False)
    parser.add_argument("--experiment", type=str, help="Experiment to run", required=True)
    parser.add_argument("--config", type=str, help="Path to json config training file", required=False)
    return parser.parse_args()

if __name__ == "__main__":
    #check_model = CheckModelWrapper(ModelTitanic, CONFIG_INSTANCE, "tests/test_data/titanic/titanic_stimulus.csv")
    #check_model.check_model()

    yaml = YamlRayConfigLoader("tests/test_model/titanic_model.yaml")
    config = yaml.get_config_instance()
    print("tested config: ", config)
    model_wrapper = CheckModelWrapper(ModelTitanic, config, "tests/test_data/titanic/titanic_stimulus.csv")
    model_wrapper.check_model()