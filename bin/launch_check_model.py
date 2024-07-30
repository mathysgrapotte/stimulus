#!/usr/bin/env python3

import argparse
import os
import json
import yaml

from launch_utils import import_class_from_file, get_experiment, memory_split_for_ray_init
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
    parser.add_argument("-w", "--initial_weights", type=str, required=False, nargs='?', const=None, default=None, metavar="FILE", help="The path to the initial weights. These can be used by the model instead of the random initialization")
    parser.add_argument("--gpus", type=int, required=False, nargs='?', const=None, default=None, metavar="NUM_OF_MAX_GPU", help="Use to limit the number of GPUs ray can use. This might be useful on many occasions, especially in a cluster system. The default value is None meaning ray will use all GPUs available. It can be set to 0 to use only CPUs.")
    parser.add_argument("--cpus", type=int, required=False, nargs='?', const=None, default=None, metavar="NUM_OF_MAX_CPU", help="Use to limit the number of CPUs ray can use. This might be useful on many occasions, especially in a cluster system. The default value is None meaning ray will use all CPUs available. It can be set to 0 to use only GPUs.")
    parser.add_argument("--memory", type=str, required=False, nargs='?', const=None, default=None, metavar="MAX_MEMORY", help="ray can have a limiter on the total memory it can use. This might be useful on many occasions, especially in a cluster system. The default value is None meaning ray will use all memory available.")
    parser.add_argument("-n", "--num_samples", type=int, required=False, nargs='?', const=3, default=3, metavar="NUM_SAMPLES", help="the config given for the tuning will have the field tune.tune_params.num_samples overwritten by this value. This means a more or less extensive representation of all possible combinations of choices for the tuning. For each run inside tune a snapshot of the config is taken and some params are chosen like loss function gradient descent, batch size etc. Some of this combination may not be compatible with either the data or the model. So the higher this value is the more likely that every value for a given param is tested. But if there are not that many choices in the tune config there is no point in putting a high value. Default is 3.")
    parser.add_argument("--ray_results_dirpath", type=str, required=False, nargs='?', const=None, default=None, metavar="DIR_PATH", help="the location where ray_results output dir should be written. if set to None (default) ray will be place it in ~/ray_results. ")
    parser.add_argument("--debug_mode", type=str, required=False, nargs='?', const=False, default=False, metavar="DEV", help="activate debug mode for tuning. default false, no debug.")

    args = parser.parse_args()
        
    return args



def main(data_path: str,
         model_path: str,
         experiment_config: str,
         config_path: str,
         initial_weights_path: str = None,
         gpus: int = None,
         cpus: int = None,
         memory: str = None,
         num_samples: int = 3,
         ray_results_dirpath: str = None,
         _debug_mode: str = False) -> None:

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
        # differentiate between schedulers, some may have different params
        # for asha make all parameters regarding the length of the tune run equal to 1
        if user_tune_config["tune"]["scheduler"]["name"] == "ASHAScheduler":
            user_tune_config["tune"]["scheduler"]["params"]["max_t"]        = 1
            user_tune_config["tune"]["scheduler"]["params"]["grace_period"] = 1
            user_tune_config["tune"]["step_size"]                           = 1
        # for the FIFO scheduler is simpler just set the stop criteria at 1 iteration
        elif user_tune_config["tune"]["scheduler"]["name"] == "FIFOScheduler":
            user_tune_config["tune"]["run_params"]["stop"]["training_iteration"] = 1

        # add initial weights to the config, when provided
        if initial_weights_path is not None:
            user_tune_config["model_params"]["initial_weights"] = os.path.abspath(initial_weights_path)

        # TODO future schedulers specific info will go here as well. maybe find a cleaner way.

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

    # compute the memory requirements for ray init. Usefull in case ray detects them wrongly. Memory is split in two for ray: for store_object memory and the other actual memory for tuning. The following function takes the total possible usable/allocated memory as a string parameter and return in bytes the values for store_memory (30% as default in ray) and memory (70%).
    object_store_mem, mem = memory_split_for_ray_init(memory)

    # set ray_result dir ubication. TODO this version of pytorch does not support relative paths, in future maybe good to remove abspath.
    ray_results_dirpath = None if ray_results_dirpath is None else os.path.abspath(ray_results_dirpath)

    # Create the learner
    learner = StimulusTuneWrapper(updated_tune_conf,
                                  model_class,
                                  downsampled_csv,
                                  initialized_experiment_class,
                                  max_gpus=gpus,
                                  max_cpus=cpus,
                                  max_object_store_mem=object_store_mem,
                                  max_mem=mem,
                                  ray_results_dir=ray_results_dirpath,
                                  _debug=_debug_mode) # TODO this version of pytorch does not support relative paths, in future maybe good to remove abspath
    
    # Tune the model and get the tuning results
    grid_results = learner.tune()

    # check that there were no errors during tuning. Tune still sends exitcode 0 even on internal errors.
    for i in range(len(grid_results)):
        result = grid_results[i]
        if not result.error:
            print(f"Trial finishes successfully with metrics" f"{result.metrics}.")
        else:
            raise TypeError(f"Trial failed with error {result.error}.")




if __name__ == "__main__":
    args = get_args()
    main(args.data, 
         args.model, 
         args.experiment, 
         args.config,
         args.initial_weights,
         args.gpus, 
         args.cpus,
         args.memory, 
         args.num_samples, 
         args.ray_results_dirpath,
         args.debug_mode)