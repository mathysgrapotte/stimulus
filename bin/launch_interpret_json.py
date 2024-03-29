#!/usr/bin/env python3

import argparse
import json
from json_schema import JsonSchema
import os


def get_args():

    """get the arguments when using from the commandline
    TODO write help function description"""
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-j", "--json", type=str, required=True, metavar="FILE", help='The json config file that hold all parameter info')
    parser.add_argument("-d", "--out_dir", type=str, required=True, metavar="DIR", help='The output dir where all he jason are written to. Output Json will be called input_json_nam-#[number].json')

    args = parser.parse_args()
    return args






def interpret_json(input_json: dict) -> list:

    # TODO handle no noise or splitter

    # Initialize json schema it checks for correctness of the Json architecture and fields / values
    schema = JsonSchema(input_json)

    # compute all noise combinations
    # first set right fucntion call based on schema.interpret_params_mode, done like following because if are inefficient
    # both function output an empty list if there is no noise argument
    function_call_dict = {"culumn_wise": schema.noise_column_wise_combination, "all_combinations": schema.noise_all_combination}
    list_noise_combinations = function_call_dict[schema.interpret_params_mode]()

    # compute all split combinations, this will only be all vs all because there is no concept of column_name, it will return empty list if there is no split function
    list_split_combinations = schema.split_combination()

    # combine split possibilities with noise ones in a all vs all manner, each splitter wil be assigned to each noiser
    list_of_json_to_write = []

    # Check if both lists are empty
    if not list_noise_combinations and not list_split_combinations:
        list_of_json_to_write.append({"experiment": schema.experiment})
    else:
        if not list_split_combinations:  # Check if list_split_combinations is empty
            for noiser_dict in list_noise_combinations:
                list_of_json_to_write.append({"experiment": schema.experiment, "noise": noiser_dict})
        else:
            for splitter_dict in list_split_combinations:
                if not list_noise_combinations:  # Check if list_noise_combinations is empty
                    list_of_json_to_write.append({"experiment": schema.experiment, "split": splitter_dict})
                else:
                    list_of_json_to_write.append({"experiment": schema.experiment, "noise": noiser_dict, "split": splitter_dict})

    # deal wiht custom if present, in this case nothing at all will be done to the dictionary, it will just be passed as it is
    for custom_dict in schema.custom_arg :
        list_of_json_to_write.append(custom_dict)

    return list_of_json_to_write

   

def main(config_json: str, out_dir_path: str) -> str:

    # open and read Json
    config = {}
    with open(config_json, 'r') as in_json:
        config = json.load(in_json)

    # interpret the json
    list_json = interpret_json(config)
    
    # write all the resultin json files
    # Create the directory if it doesn't exist
    os.makedirs(out_dir_path, exist_ok=True)

    # Populate the directory with files containing the single SJon combination
    for i, elements in enumerate(list_json):
        suffix = os.path.splitext(os.path.basename(config_json))[0]
        file_path = os.path.join(out_dir_path, f"{suffix}-#{i+1}.json")
        with open(file_path, 'w') as file:
            file.write(f"{elements}\n")


if __name__ == "__main__":
    args = get_args()
    main(args.json, args.out_dir)