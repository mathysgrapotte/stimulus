#!/usr/bin/env python3

import argparse
import json

from launch_utils import get_experiment
from src.data.csv import CsvProcessing



def get_args():

    "get the arguments when using from the commandline"
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--csv", type=str, required=True, metavar="FILE", help='The file path for the csv containing all data')
    parser.add_argument("-j", "--json", type=str, required=True, metavar="FILE", help='The json config file that hold all parameter info')
    parser.add_argument("-o", "--output", type=str, required=True, metavar="FILE", help='The output file path to write the noised csv')

    args = parser.parse_args()
    return args


def main(data_csv, config_json, out_path):
    """
    This launcher will be the connection between the csv and one json configuration.
    It should also handle some sanity checks.
    """
    
    # open and read Json
    config = {}
    with open(config_json, 'r') as in_json:
        config = json.load(in_json)

    # initialize the experiment class
    exp_obj = get_experiment(config["experiment"])

    # initialize the csv processing class, it open and reads the csv in automatic 
    csv_obj = CsvProcessing(exp_obj, data_csv)
    
    # Transform the data according to what defined in the experiment class and the specifics of the user in the Json
    # in case of no transformation specification so when the config has "augmentation" : None  just save a copy of the original csv file
    if config.get("transform") is not None:
        csv_obj.transform(config["transform"])

    # save the modified csv
    csv_obj.save(out_path)


if __name__ == "__main__":
    args = get_args()
    main(args.csv, args.json, args.output)
