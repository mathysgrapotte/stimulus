#!/usr/bin/env python3

import argparse
from bin.src.learner.raytune_learner import TuneTrainWrapper as StimulusTuneTrainWrapper
import ray.tune as tune
from ray import train, tune
import ray.tune.schedulers as schedulers
import os

def get_args():

    "get the arguments when using from the commandline"
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--config", type=str, required=True, metavar="FILE", help='The file path for the config file')
    parser.add_argument("-m", "--model", type=str, required=True, metavar="FILE", help='The model file')
    parser.add_argument("-d", "--data", type=str, required=True, metavar="FILE", help='The data file')
    parser.add_argument("-e", "--experiment", type=str, required=True, metavar="FILE", help='The path to the experiment')
    parser.add_argument("-o", "--output", type=str, required=True, metavar="FILE", help='The output file path to write the trained model to')
    parser.add_argument("-bc", "--best_config", type=str, required=True, metavar="FILE", help='The path to write the best config to')

    args = parser.parse_args()
    return args




def main(config_path, model_path, data_path, experiment_path, output, best_config_path):
    """
    This launcher use ray tune to find the best hyperparameters for a given model.
    """

    # Create the learner
    learner = StimulusTuneTrainWrapper(config_path, model_path, experiment_path, data_path)
    
    # Tune the model
    learner.tune()
    
    # save best config
    learner.save_best_config(best_config_path)

    # Train the model with the best config
    learner.train()

    # Save the model
    learner.trainer.export_model(output)


if __name__ == "__main__":
    args = get_args()
    main(args.config, args.model, args.data, args.experiment, args.output, args.best_config)
