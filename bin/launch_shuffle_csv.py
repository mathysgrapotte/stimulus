#!/usr/bin/env python3

import argparse
import json
import os
from src.data.csv import CsvProcessing
import src.data.experiments as exp


def get_args():

    "get the arguments when using from the commandline"
    
    parser = argparse.ArgumentParser(description="TODO fill this description")
    parser.add_argument("-c", "--csv", type=str, required=True, metavar="FILE", help='The file path for the csv containing all data')
    parser.add_argument("-j", "--json", type=str, required=True, metavar="FILE", help='The json config file that hold all parameter info')
    parser.add_argument("-o", "--output", type=str, required=True, metavar="FILE", help='The output file path to write the noised csv')

    args = parser.parse_args()
    return args




def main(data_csv, config_json, out_path):
    """
    This scripts shuffles the data and then splits it acording to the default split method, most likely RandomSplit.

    TODO major changes when this is going to select a given shuffle method and integration with split.
    """
    
    # open and read Json, jsut to extract the experiment name, so all other fields are scratched
    config = {}
    with open(config_json, 'r') as in_json:
        tmp = json.load(in_json)
        config["experiment"] = tmp["experiment"]
        config["split"] = {"name": "RandomSplitter", "params": {}}

    # write the config modified, this will be associated to the shuffled data. TODO better solution to renaming like this
    modified_json = os.path.splitext(os.path.basename(data_csv))[0] + '-shuffled.json'
    with open(modified_json, 'w') as out_json:
        json.dump(config, out_json)

    # initialize the experiment class
    exp_obj = getattr(exp, config["experiment"])() 

    # initialize the csv processing class, it open and reads the csv in automatic 
    csv_obj = CsvProcessing(exp_obj, data_csv)

    # shuffle the data
    csv_obj.shuffle_labels()

    # split the data
    # split column already present in csv , override it with random split (default splitter)
    # TODO change this behaviour to do both, maybe
    csv_obj.add_split(config["split"], force = True)

    # save the modified csv
    csv_obj.save(out_path)





if __name__ == "__main__":
    args = get_args()
    main(args.csv, args.json, args.output)