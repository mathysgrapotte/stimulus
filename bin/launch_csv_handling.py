#!/usr/bin/env python3

import argparse
from src.data.csv_parser import CSVParser
import src.data.experiments as exp


def get_args():

        "get the arguments when using from the commandline"

        parser = argparse.ArgumentParser(description="")
        parser.add_argument("-c", "--csv", type=str, required=True, metavar="FILE", help='The file path for the csv containing all data')
        parser.add_argument("-j", "--json", type=str, required=True, metavar="FILE", help='The json config file that hold all parameter info')

        args = parser.parse_args()
        return args


   

def main(data_csv, config_json):

    print(data_csv, config_json)


    """
    experiment = exp.eval(json[experiment_name])
    data = CsvHandler( data_csv, experiment )
    data.noise(json[params])
    data.split(jason[splitparamates])
    """



if __name__ == "__main__":
        args = get_args()
        main(args.csv, args.json)