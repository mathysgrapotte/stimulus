#!/usr/bin/env python3

import argparse
import json
from json_schema import JsonSchema
import os


def get_args():

    """get the arguments when using from the commandline
    
    This script reads a Json with very defined structure and creates all the Json files ready to be passed to 
    the stimulus package. 
    
    The structure of the Json is described here -> TODO paste here link to documentation.
    This Json and it's structure summarize how to generate all the transform - split and respective parameters combinations.
    Each resulting Json will hold only one combination of the above three things.

    This script will always generate at least on Json file that represent the combination that does not touch the data (no transform)
    and uses the defalut split behaviour.
    """
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-j", "--json", type=str, required=True, metavar="FILE", help='The json config file that hold all transform - split - parameter info')
    parser.add_argument("-d", "--out_dir", type=str, required=True, metavar="DIR", help='The output dir where all he jason are written to. Output Json will be called input_json_nam-#[number].json')

    args = parser.parse_args()
    return args






def interpret_json(input_json: dict) -> list:

    # Initialize json schema it checks for correctness of the Json architecture and fields / values
    schema = JsonSchema(input_json)

    # compute all transform combinations
    # first set right fucntion call based on schema.interpret_params_mode, done like following because if are inefficient
    # both function output an empty list if there is no transform argument
    function_call_dict = {"column_wise": schema.transform_column_wise_combination, "all_combinations": schema.transform_all_combination}
    
    # if transform is not present no need to compute the list of possibilities
    list_transform_combinations = [None]
    if schema.transform_arg:
        list_transform_combinations = function_call_dict[schema.interpret_params_mode]()
        
    # if split is present, again like above
    list_split_combinations = [None]
    if schema.split_arg:
        # compute all split combinations, this will only be all vs all because there is no concept of column_name, it will return empty list if there is no split function
        list_split_combinations = schema.split_combination()

    # combine split possibilities with transform ones in a all vs all manner, each splitter wil be assigned to each transformed
    list_of_json_to_write = []

    # The  pipeline has always to happen at least twice, aka on the data itself untouched (line below) and on data with labels shuffled. This line is not necessary only in the case of missing both transform and spli arguments in the inpèut Json.
    if schema.transform_arg or schema.split_arg:
        list_of_json_to_write.append({"experiment": schema.experiment, "transform": None, "split": None})
         
    # The following lines generate all the ready to write json dictionaries, combining all vs all the transformeds combination with the splitter combinations
    for transformed_dict in list_transform_combinations:
        for splitter_dict in list_split_combinations:
            list_of_json_to_write.append({"experiment": schema.experiment, "transform": transformed_dict, "split": splitter_dict})

    # deal wiht custom if present, in this case nothing at all will be done to the dictionaries present in the list except adding the experiment name to it. The user is responsible for the dict inside custom to be correct and ready for the csv_launcher
    for custom_dict in schema.custom_arg :
        new_dict = {**{"experiment": schema.experiment}, **custom_dict}
        list_of_json_to_write.append(new_dict)


    return list_of_json_to_write

   

def main(config_json: str, out_dir_path: str) -> str:

    # open and read Json
    config = {}
    with open(config_json, 'r') as in_json:
        config = json.load(in_json)

    # interpret the json
    list_json = interpret_json(config)
    
    # write all the resulting json files
    # create the directory if it doesn't exist
    os.makedirs(out_dir_path, exist_ok=True)

    # Populate the directory with files containing the single SJon combination
    for i, interpreted_json in enumerate(list_json):
        suffix = os.path.splitext(os.path.basename(config_json))[0]
        file_path = os.path.join(out_dir_path, f"{suffix}-#{i+1}.json")
        with open(file_path, 'w') as json_file:
            json.dump(interpreted_json, json_file)


if __name__ == "__main__":
    args = get_args()
    main(args.json, args.out_dir)
