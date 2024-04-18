#!/usr/bin/env python3

import argparse
from bin.src.learner.raytune_learner import TuneModel as RayTuneLearner
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
    parser.add_argument("-o", "--output", type=str, required=True, metavar="FILE", help='The output file path to write the trained model to')
    parser.add_argument("-bc", "--best_config", type=str, required=True, metavar="FILE", help='The path to write the best config to')

    args = parser.parse_args()
    return args




def main(config, model, output, best_config_path):
    """
    This launcher use ray tune to find the best hyperparameters for a given model.

    TODO this is not complete - finish!
    """
    
    # TODO Add right data path to the config and model path!



    # Load the config
    config = {}
    with open("bin/tests/test_model/simple.config", "r") as f:
        config = eval(f.read())


    # Get the scheduler from the config
    try:
        scheduler = getattr(schedulers, config["scheduler"]["name"])( **config["scheduler"]["params"])
    except AttributeError:
        raise ValueError(f"Invalid optimizer: {config['scheduler']['name']}, check PyTorch for documentation on available optimizers")

    # Get the tune config
    try: 
        tune_config = getattr(tune, config["tune_config"]["name"])(scheduler = scheduler, **config["tune_config"]["params"])
    except AttributeError:
        raise ValueError(f"Invalid tune_config: {config['tune_config']['name']}, check PyTorch for documentation on how a tune_config should be defined")

    # Get checkpoint config 
    try:
        checkpoint_config = getattr(train, config["checkpoint_config"]["name"])(**config["checkpoint_config"]["params"])
    except AttributeError:
        raise ValueError(f"Invalid checkpoint_config: {config['checkpoint_config']['name']}, check PyTorch for documentation on how a checkpoint_config should be defined")

    # Get the run config
    try:
        run_config = getattr(train, config["run_config"]["name"])(checkpoint_config = checkpoint_config, **config["run_config"]["params"])
    except AttributeError:
        raise ValueError(f"Invalid run_config: {config['run_config']['name']}, check PyTorch for documentation on how a run_config should be defined")

    # Set tuner 
    tuner = tune.Tuner(RayTuneLearner,
                        tune_config= tune_config,
                        param_space=config,
                        run_config=run_config,
                    )
    
    # Run hyperparameter tuning   
    results = tuner.fit()

    # Obtain best config
    best_result = results.get_best_result()
    best_config = os.path.join(best_result.path, "params.json")
    best_config = eval(open(best_config, "r").read())
    
    # save best config
    with open(best_config_path, "w") as f:
        f.write(str(best_config))






if __name__ == "__main__":
    args = get_args()
    main(args.csv, args.json, args.output)
