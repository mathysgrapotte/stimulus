#!/usr/bin/env python3

import argparse
import json

from launch_utils import import_class_from_file, get_experiment
from src.learner.raytune_learner import TuneWrapper as StimulusTuneWrapper
from src.learner.raytune_parser import TuneParser as StimulusTuneParser
from json_schema import JsonSchema

def get_args():

    "get the arguments when using from the commandline"
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--config", type=str, required=True, metavar="FILE", help='The file path for the config file')
    parser.add_argument("-m", "--model", type=str, required=True, metavar="FILE", help='The model file')
    parser.add_argument("-d", "--data", type=str, required=True, metavar="FILE", help='The data file')
    parser.add_argument("-e", "--experiment_config", type=str, required=True, metavar="FILE", help='The json used to modify the data. Inside it has the experiment name as specified in the experimets.py, this will then be dinamically imported during training. It is necessary to recover how the user specified the encoding of the data. Data is encoded on the fly.')
    parser.add_argument("-o", "--output", type=str, required=False,  nargs='?', const='best_model.pt', default='best_model.pt', metavar="FILE", help='The output file path to write the trained model to')
    parser.add_argument("-bc", "--best_config", type=str, required=False, nargs='?', const='best_config.json', default='best_config.json', metavar="FILE", help='The path to write the best config to')
    parser.add_argument("-bm", "--best_metrics", type=str, required=False, nargs='?', const='best_metrics.csv', default='best_metrics.csv', metavar="FILE", help='The path to write the best metrics to')
    parser.add_argument("-bo", "--best_optimizer", type=str, required=False, nargs='?', const='best_optimizer.pt', default='best_optimizer.pt', metavar="FILE", help='The path to write the best optimizer to')

    args = parser.parse_args()
    return args

def main(config_path: str, model_path: str, data_path: str, experiment_config: str, output: str, best_config_path: str, best_metrics_path: str, best_optimizer_path: str) -> None:
    """
    This launcher use ray tune to find the best hyperparameters for a given model.
    """

    # TODO update to yaml the experimnt config
    # load json into dictionary
    exp_config = {}
    with open(experiment_config, 'r') as in_json:
        exp_config = json.load(in_json)

    # initialize the experiment class
    initialized_experiment_class = get_experiment(exp_config["experiment"])

    # import the model correctly but do not initialize it yet, ray_tune does that itself
    model_class = import_class_from_file(model_path)

    # Create the learner
    learner = StimulusTuneWrapper(config_path, model_class, data_path, initialized_experiment_class)
    
    # Tune the model and get the tuning results
    results = learner.tune()

    # parse raytune results
    results = StimulusTuneParser(results)
    results.save_best_model(output)
    results.save_best_config(best_config_path)
    results.save_best_metrics_dataframe(best_metrics_path)
    results.save_best_optimizer(best_optimizer_path)


if __name__ == "__main__":
    args = get_args()
    main(args.config, args.model, args.data, args.experiment_config, args.output, args.best_config, args.best_metrics, args.best_optimizer)
