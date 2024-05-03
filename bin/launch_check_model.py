import argparse
import torch.nn as nn
import torch
import os
import src.data.handlertorch as handlertorch

from copy import deepcopy
from src.utils.yaml_model_schema import YamlRayConfigLoader
from launch_utils import import_class_from_file, get_experiment
    
class CheckModelWrapper:
    def __init__(self, model: nn.Module, config_instance: dict, data_path: str, experiment: object):
        self.model = model(**config_instance["model_params"])
        # get the optimizer from pytorch
        optimizer = getattr(torch.optim, config_instance["optimizer"]["method"])
        # get the loss function from pytorch
        self.loss_fn = getattr(torch.nn, config_instance["loss_params"]["loss_fn"])()
        # instantiate the optimizer
        self.optimizer = optimizer(self.model.parameters(), **config_instance["optimizer_params"])
        # train_data is a TorchDataset object
        self.train_data = torch.utils.data.DataLoader(handlertorch.TorchDataset(os.path.abspath(data_path), experiment, split=None), batch_size=2, shuffle=True)

    def check_model(self):
        # get the initial model weights into a variable 
        initial_model_weights = deepcopy(self.model.state_dict())
        # load one sample of the data

        x, y, meta = next(iter(self.train_data))
        # train the model for one epoch
        loss = self.model.batch(x, y, self.loss_fn, self.optimizer)
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

    args = arg_parser()
    Model = import_class_from_file(args.model)
    experiment = get_experiment(args.experiment)

    yaml_config = YamlRayConfigLoader(config_path=args.config)
    config_instance = yaml_config.get_config_instance()
    print("tested config: ", config_instance)
    model_wrapper = CheckModelWrapper(Model, config_instance, args.input, experiment)
    model_wrapper.check_model()