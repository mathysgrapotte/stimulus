#!/usr/bin/env python3

import argparse
import torch.nn as nn
import torch
import os
import src.data.handlertorch as handlertorch
import json

from copy import deepcopy
from src.utils.yaml_model_schema import YamlRayConfigLoader
from launch_utils import import_class_from_file, get_experiment
from json_schema import JsonSchema



def get_args():

    """get the arguments when using from the commandline"""

    parser = argparse.ArgumentParser(description="Launch check_model")
    parser.add_argument("-d", "--data", type=str, required=True, metavar="FILE", help="Path to input csv file")
    parser.add_argument("-m", "--model", type=str, required=True, metavar="FILE", help="Path to model file")
    parser.add_argument("-e", "--experiment", type=str, required=True, metavar="FILE", help="Experiment config file. From this the experiment class name is extracted.")
    parser.add_argument("-c", "--config", type=str, required=True, metavar="FILE", help="Path to yaml config training file")
    
    args = parser.parse_args()
    return args


class CheckModelWrapper():

    def __init__(self, model: nn.Module, config_instance: dict, data_path: str, experiment: object):
        self.model = model(**config_instance["model_params"])
        # get the optimizer from pytorch
        optimizer = getattr(torch.optim, config_instance["optimizer_params"]["method"])
        # get the loss function from pytorch
        self.loss_fn = getattr(torch.nn, config_instance["loss_params"]["loss_fn"])()
        # instantiate the optimizer, get all optimizer parameters except the names of the optimizers themselves
        optimizer_params_values = {key: value for key, value in config_instance["optimizer_params"].items() if key != "method"}
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params_values)
        # train_data is a TorchDataset object
        self.train_data = torch.utils.data.DataLoader(handlertorch.TorchDataset(os.path.abspath(data_path), experiment, split=None), batch_size=2, shuffle=True)

    def check_model(self):
        # get the initial model weights into a variable 
        initial_model_weights = deepcopy(self.model.state_dict())
        # load one sample of the data

        x, y, meta = next(iter(self.train_data))
        # train the model for one epoch
        loss, output = self.model.batch(x, y, self.loss_fn, self.optimizer)
        # check the model weights have changed, and print if it has
        for key in initial_model_weights:
            if torch.equal(initial_model_weights[key], self.model.state_dict()[key]):
                print(f"Model weights have not changed for key {key}")
            else:
                print(f"Model weights have changed for key {key}")

        # print the computed loss, displaying loss_fn name as well
        print(f"Loss computed with {self.loss_fn.__class__.__name__} is {loss.item()}")


def main(data: str, model_file: str, experiment_config: str, config_file: str):

    # TODO update to yaml the experimnt config
    # load json into dictionary
    exp_config = {}
    with open(experiment_config, 'r') as in_json:
        exp_config = json.load(in_json)
    
    # Initialize json schema it checks for correctness of the Json architecture and fields / values. already raises errors.
    schema = JsonSchema(exp_config)

    Model = import_class_from_file(model_file)
    experiment = get_experiment(schema.experiment)

    yaml_config = YamlRayConfigLoader(config_path=config_file)
    config_instance = yaml_config.get_config_instance()
    print("tested config: ", config_instance)
    model_wrapper = CheckModelWrapper(Model, config_instance, data, experiment)
    model_wrapper.check_model()


if __name__ == "__main__":
    args = get_args()
    main(args.data, args.model, args.experiment, args.config)