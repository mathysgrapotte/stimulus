#!/usr/bin/env python3

import argparse
import json
import os
import importlib.util

from src.learner.raytune_learner import TuneTrainWrapper as StimulusTuneTrainWrapper
from launch_utils import import_class_from_file, get_experiment
from src.utils.yaml_model_schema import YamlRayConfigLoader

def get_args():

    "get the arguments when using from the commandline"
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--config", type=str, required=True, metavar="FILE", help='The file path for the config file')
    parser.add_argument("-m", "--model", type=str, required=True, metavar="FILE", help='The model file')
    parser.add_argument("-d", "--data", type=str, required=True, metavar="FILE", help='The data file')
    parser.add_argument("-j", "--json_experiment", type=str, required=True, metavar="FILE", help='The json used to modify the data. Inside it has the experiment name as specified in the experimets.py, this will then be dinamically imported during training. It is necessary to recover how the user specified the encoding of the data. Data is encoded on the fly.')
    parser.add_argument("-o", "--output", type=str, required=False,  nargs='?', const='best_model.pt', default='best_model.pt', metavar="FILE", help='The output file path to write the trained model to')
    parser.add_argument("-bc", "--best_config", type=str, required=False, nargs='?', const='best_config.json', default='best_config.json', metavar="FILE", help='The path to write the best config to')

    args = parser.parse_args()
    return args

def main(config_path: str, model_path: str, data_path: str, json_experiment: str, output: str, best_config_path: str) -> None:
    """
    This launcher use ray tune to find the best hyperparameters for a given model.
    """

    # import the model correctly but do not initialize it yet, ray_tune does that itself
    model_class = import_class_from_file(model_path)

    # read json and retrieve experiment name and then initialize the experiment class
    experiment_name = None
    with open(json_experiment, 'r') as in_json:
        d = json.load(in_json)
        experiment_name = d["experiment"]
    initialized_experiment_class = get_experiment(experiment_name)

    # Create the learner
    learner = StimulusTuneTrainWrapper(config_path, model_class, data_path, initialized_experiment_class)
    
    # Tune the model
    learner.tune()
    
    # save best config
    learner.store_best_config(best_config_path)

    # TODO report best model

    # Train the model with the best config and best model, aka fine-tuning
    #learner.train()
    # Save the model fine-tuned model
    #learner.trainer.export_model(output)    

if __name__ == "__main__":
    args = get_args()
    main(args.config, args.model, args.data, args.json_experiment, args.output, args.best_config)
