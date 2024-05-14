#!/usr/bin/env python3

import argparse
import os
import json
import yaml

from launch_utils import import_class_from_file, get_experiment
from json_schema import JsonSchema
from src.learner.raytune_learner import TuneWrapper as StimulusTuneWrapper
from src.data.csv import CsvProcessing


def get_args():

    """get the arguments when using from the commandline"""

    parser = argparse.ArgumentParser(description="Launch check_model")
    parser.add_argument("-d", "--data", type=str, required=True, metavar="FILE", help="Path to input csv file")
    parser.add_argument("-m", "--model", type=str, required=True, metavar="FILE", help="Path to model file")
    parser.add_argument("-e", "--experiment", type=str, required=True, metavar="FILE", help="Experiment config file. From this the experiment class name is extracted.")
    parser.add_argument("-c", "--config", type=str, required=True, metavar="FILE", help="Path to yaml config training file")
    parser.add_argument("-n", "--num_samples", type=int, required=False, nargs='?', const=3, default=3, metavar="TUNE_PARAM", help="the config given for the tuning will have the field tune.tune_params.num_samples overwritten by this value. This means a more or less extensive representation of all possible combination of choiches for the tuning. For each run inside tune a snapshot of the config is taken and some params are chosen like loss function gradient descent, batch size ecc.. . Some of this combination may not be compatible with either the data or the model. So the higher this value is the most likely that every value for a given param is tested. But if there are not that many choiches in the tune config there is no point in putting an high value. Default is 3.")

    args = parser.parse_args()
    return args

"""
class CheckModelWrapper():

    def __init__(self, model: nn.Module, config_file: str, data_path: str, experiment: object):
        # open the yaml tune config file and get an instance of it . aka choose values when needed.
        yaml_config = YamlRayConfigLoader(config_file)
        config_instance = yaml_config.get_config_instance()
        print("tested config: ", config_instance)
        self.model = model(**config_instance["model_params"])
        # get the optimizer from pytorch
        optimizer = getattr(torch.optim, config_instance["optimizer_params"]["method"])
        # get the loss dict from the config and replace the each value with the actual imported/initialized loss dfunction 
        self.loss_dict = yaml_config.initialize_loss_functions(config_instance["loss_params"])
        # instantiate the optimizer, get all optimizer parameters except the names of the optimizers themselves
        optimizer_params_values = {key: value for key, value in config_instance["optimizer_params"].items() if key != "method"}
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params_values)
        # train_data is a TorchDataset object
        self.train_data = torch.utils.data.DataLoader(handlertorch.TorchDataset(os.path.abspath(data_path), experiment, split=None), batch_size=config_instance["data_params"]["batch_size"], shuffle=True)

    def check_model(self):
        # get the initial model weights into a variable 
        initial_model_weights = deepcopy(self.model.state_dict())
        # load one sample of the data

        x, y, meta = next(iter(self.train_data))
        # train the model for one epoch
        loss, output = self.model.batch(x, y, self.loss_dict, self.optimizer)
        # check the model weights have changed, and print if it has
        for key in initial_model_weights:
            if torch.equal(initial_model_weights[key], self.model.state_dict()[key]):
                print(f"Model weights have not changed for key {key}")
            else:
                print(f"Model weights have changed for key {key}")

        # print the computed loss, displaying loss_dict as well
        print(f"Loss computed with {self.loss_dict} is {loss.item()}")
"""

def main(data_path: str, model_path: str, experiment_config: str, config_path: str, num_samples: int):

    # TODO update to yaml the experimnt config
    # load json into dictionary
    exp_config = {}
    with open(experiment_config, 'r') as in_json:
        exp_config = json.load(in_json)

    # Initialize json schema it checks for correctness of the Json architecture and fields / values. already raises errors.
    schema = JsonSchema(exp_config)

    # initialize the experiment class
    initialized_experiment_class = get_experiment(schema.experiment)

    # import the model correctly but do not initialize it yet, ray_tune does that itself
    model_class = import_class_from_file(model_path)

    # Update the tune config file. no need to run the whole amount asked for the tuning. Basically downsample the tuning.
    updated_tune_conf = "check_model_modified_tune_config.yaml"
    with open(config_path, 'r') as conf_file, open(updated_tune_conf, "w") as new_conf:
        user_tune_config = yaml.safe_load(conf_file)
        # make so the tune run just once per num_sample
        user_tune_config["tune"]["tune_params"]["num_samples"]          = num_samples
        user_tune_config["tune"]["scheduler"]["params"]["max_t"]        = 1
        user_tune_config["tune"]["scheduler"]["params"]["grace_period"] = 1
        user_tune_config["tune"]["step_size"]                           = 1

        # TODO check if among the first 2 values of all splitters params there is a percentage that makes the resulting split smaller that the biggest batch value

        # save to file the new dictionary because StimulusTuneWrapper only takes paths
        yaml.dump(user_tune_config, new_conf)

    # initialize the csv processing class, it open and reads the csv in automatic 
    csv_obj         = CsvProcessing(initialized_experiment_class, data_path)
    downsampled_csv = "downsampled.csv"

    # TODO downsample data, do so tasking care of batch size so good point to tell user batch size is too big



    # add the split column if not present
    if "split" not in csv_obj.check_and_get_categories():
        # split values are set to be half the data given so that the downsampled file total lines can be as little as possible
        config_default = {"name": "RandomSplitter", "params": {"split": [0.5, 0.5, 0.0]} }
        csv_obj.add_split(config_default)
    
    # save the modified csv
    csv_obj.save(downsampled_csv)


    # Create the learner
    learner = StimulusTuneWrapper(updated_tune_conf, model_class, downsampled_csv, initialized_experiment_class)
    
    # Tune the model and get the tuning results
    results = learner.tune()

    # check that there were no errors during tuning. Tune still sends exitcode 0 even on internal errors.
    for i in range(len(results)):
        result = results[i]
        if not result.error:
            print(f"Trial finishes successfully with metrics" f"{result.metrics}.")
        else:
            raise TypeError(f"Trial failed with error {result.error}.")




if __name__ == "__main__":
    args = get_args()
    main(args.data, args.model, args.experiment, args.config, args.num_samples)