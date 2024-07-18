#!/usr/bin/env python3

import argparse
import json
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.handlertorch import TorchDataset
from src.learner.predict import PredictWrapper
from launch_utils import import_class_from_file, get_experiment

def get_args():
    
    "get the arguments when using from the commandline"
    
    parser = argparse.ArgumentParser(description="Predict the output of a model on a dataset.")
    parser.add_argument("-m", "--model", type=str, required=True, metavar="FILE", help='The model .py file')
    parser.add_argument("-w", "--weight", type=str, required=True, metavar="FILE", help="Model weights .pt file")
    parser.add_argument("-mc", "--model_config", type=str, required=True, metavar="FILE", help="The tune config file.")
    parser.add_argument("-ec", "--experiment_config", type=str, required=True, metavar="FILE", help='The experiment config used to modify the data.')
    parser.add_argument("-d", "--data", type=str, required=True, metavar="FILE", help='Input data')
    parser.add_argument("-o", "--output", type=str, required=True, metavar="FILE", help="output predictions csv file")
    parser.add_argument("--split", type=int, help="The split of the data to use (default: None)")
    parser.add_argument("--return_labels", type=bool, default=True, help="return the labels with the prediction (default: True)")

    args = parser.parse_args()
    return args

def main(model_path: str, weight_path: str, mconfig_path: str, econfig_path: str, data_path: str, output: str, return_labels: bool = False):
    
    # load model
    model_class = import_class_from_file(model_path)
    model = load_model( model_class, weight_path, mconfig_path )

    # read experiment config and retrieve experiment name and then initialize the experiment class
    experiment_name = None
    with open(econfig_path, 'r') as in_json:
        d = json.load(in_json)
        experiment_name = d["experiment"]
    initialized_experiment_class = get_experiment(experiment_name)

    # load and encode data
    data = pd.read_csv(data_path)
    dataloader = DataLoader(TorchDataset(data_path, initialized_experiment_class, split=split), batch_size=256, shuffle=False)  # TODO batch size is hard coded here, but maybe in the future we change it

    # predict
    out = PredictWrapper(model, dataloader).predict(return_labels=return_labels)
    if return_labels:
        y_pred, y_true = out
    else:
        y_pred, y_true = out, {}
    
    # make the keys coherent with the columns from input data csv
    y_pred = parse_y_keys(y_pred, data, y_type='pred')
    y_true = parse_y_keys(y_true, data, y_type='label')

    # parse predictions (and labels) into a data frame
    y = {**y_pred, **y_true}
    df = pd.DataFrame(y)
    df = add_meta_info(data, df)

    # save output 
    df.to_csv(output, index=False)

def load_model(model_class: object, weight_path: str, mconfig_path: str) -> object:
    """
    Load the model with its config and weights.
    """
    # load model config
    with open(mconfig_path, 'r') as in_json:
        mconfig = json.load(in_json)["model_params"]

    # load model
    model = model_class(**mconfig)
    model.load_state_dict(torch.load(weight_path))

    return model

def parse_y_keys(y: dict, data: pd.DataFrame, y_type='pred'):
    """
    Parse the keys of the y dictionary.

    Basically, it replaces the keys of the y dictionary with the keys of the input data.
    such as 'binding' by 'binding:pred:float' or 'binding:label:float'
    """
    # return if empty
    if len(y) == 0:
        return y

    # get the keys
    keys_y = y.keys()
    keys_data = data.columns

    # parse the dictionary with the new keys
    parsed_y = {}
    for k1 in keys_y:
        for k2 in keys_data:
            if k1 == k2.split(':')[0]:
                new_key = f"{k1}:{y_type}:{k2.split(':')[2]}"
                parsed_y[new_key] = y[k1]

    return parsed_y

def add_meta_info(data: pd.DataFrame, df: pd.DataFrame):
    """
    Add the meta columns to the data frame of predictions.
    In this way the output file can also contain the meta information.
    """
    # get meta keys
    keys = get_meta_keys(data.columns)

    # add meta info
    for key in keys:
        df[key] = data[key]

    return df

def get_meta_keys(names: list):
    """
    Get the `meta` column keys.
    """
    keys = []
    for name in names:
        fields = name.split(":")
        if fields[1] == "meta":
            keys.append(name)
    return keys

if __name__ == "__main__":
    args = get_args()
    main(args.model, args.weight, args.model_config, args.experiment_config, args.data, args.output, args.return_labels)