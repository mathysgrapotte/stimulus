#!/usr/bin/env python3

import argparse
import json
from json_schema import JsonSchema


def get_args():

    "get the arguments when using from the commandline"
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-j", "--json", type=str, required=True, metavar="FILE", help='The json config file that hold all parameter info')
    
    args = parser.parse_args()
    return args






def interpret_json(input_json: dict) -> list:

    # TODO handle no noise or splitter

    # Initialize json schema it checks for correctness of the Json architecture and fields / values
    schema = JsonSchema(input_json)

    #print("\nnoise_configurations :\n", schema.noise_arg, "\n", type(schema.noise_arg))
    #print("\nsplit_configurations :\n", schema.split_arg, "\n", type(schema.split_arg))
    #print("\ncustom_configurations :\n", schema.custom_arg, "\n", type(schema.custom_arg))
    #print(schema.number_culumn_wise_val)
    #print(schema.experiment, schema.interpret_params_mode, schema.column_names)

    # compute all noise combinations
    # first set right fucntion call based on schema.interpret_params_mode, done like following because if are inefficient
    function_call_dict = {"culumn_wise": schema.noise_column_wise_combination, "all_combinations": schema.noise_all_combination}
    list_noise_combinatioons = function_call_dict[schema.interpret_params_mode]()

    
   

def main(config_json: str) -> str:

    # open and read Json
    config = {}
    with open(config_json, 'r') as in_json:
        config = json.load(in_json)

    # initialize the json scheme class 
    interpret_json(config)


   


if __name__ == "__main__":
    args = get_args()
    main(args.json)