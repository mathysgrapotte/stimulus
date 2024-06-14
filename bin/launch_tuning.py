#!/usr/bin/env python3

import argparse
import json
import os
import yaml

from launch_utils import import_class_from_file, get_experiment, memory_split_for_ray_init
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
    parser.add_argument("--gpus", type=int, required=False, nargs='?', const=None, default=None, metavar="NUM_OF_MAX_CPU", help="ray can have a limiter on the number of GPUs it can use. This might be usefull in many occasions, especially on a cluster system. The default value is None meaning ray will use all CGPUs available. It can be set to 0 to use only CPUs.")
    parser.add_argument("--cpus", type=int, required=False, nargs='?', const=None, default=None, metavar="NUM_OF_MAX_CPU", help="ray can have a limiter on the number of CPUs it can use. This might be usefull in many occasions, especially on a cluster system. The default value is None meaning ray will use all CPUs available. It can be set to 0 to use only GPUs.")
    parser.add_argument("--memory", type=str, required=False, nargs='?', const=None, default=None, metavar="NUM_OF_MAX_CPU", help="ray can have a limiter on the total memory it can use. This might be usefull in many occasions, especially on a cluster system. The default value is None meaning ray will use all memory available.")
    parser.add_argument("--ray_results_dirpath", type=str, required=False, nargs='?', const=None, default=None, metavar="DIR_PATH", help="the location where ray_results output dir should be written. if set to None (default) ray will be place it in ~/ray_results ")

    args = parser.parse_args()
        
    return args

def main(config_path: str,
         model_path: str,
         data_path: str,
         experiment_config: str,
         output: str,
         best_config_path: str,
         best_metrics_path: str,
         best_optimizer_path: str,
         gpus: int = None,
         cpus: int = None,
         memory: str = None,
         ray_results_dirpath: str = None) -> None:
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

    # Update the tune config file. Because if resources are specified for cpu and gpu they are overwritten with what nextflow has otherwise this field is created
    updated_tune_conf = "check_model_modified_tune_config.yaml"
    with open(config_path, 'r') as conf_file, open(updated_tune_conf, "w") as new_conf:
        user_tune_config = yaml.safe_load(conf_file)
        
        # save to file the new dictionary because StimulusTuneWrapper only takes paths
        yaml.dump(user_tune_config, new_conf)

    # compute the memory requirements for ray init. Usefull in case ray detects them wrongly. Memory is split in two for ray: for store_object memory and the other actual memory for tuning. The following function takes the total possible usable/allocated memory as a string parameter and return in bytes the values for store_memory (30% as default in ray) and memory (70%).
    object_store_mem, mem = memory_split_for_ray_init(memory)

    # Create the learner
    learner = StimulusTuneWrapper(updated_tune_conf,
                                  model_class,
                                  data_path,
                                  initialized_experiment_class,
                                  max_gpus=gpus,
                                  max_cpus=cpus,
                                  max_object_store_mem=object_store_mem,
                                  max_mem=mem,
                                  ray_results_dir=os.path.abspath(ray_results_dirpath))  # TODO this version of pytorch does not support relative paths, in future maybe good to remove abspath
    
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
    main(args.config, 
         args.model, 
         args.data, 
         args.experiment_config, 
         args.output, 
         args.best_config, 
         args.best_metrics, 
         args.best_optimizer,
         args.gpus, 
         args.cpus,
         args.memory,
         args.ray_results_dirpath)
