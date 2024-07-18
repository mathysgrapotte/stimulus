#!/usr/bin/env python3

import argparse
import json
import os
import polars as pl
import torch
from torch.utils.data import DataLoader
from typing import Tuple

from src.data.handlertorch import TorchDataset
from src.learner.predict import PredictWrapper
from launch_utils import import_class_from_file, get_experiment

def get_args():
    
    "get the arguments when using from the commandline"
    
    parser = argparse.ArgumentParser(description="Predict the output of a model on a dataset.")
    parser.add_argument("-m", "--model", type=str, required=True, metavar="FILE", help='The model .py file')
    parser.add_argument("-w", "--weight", type=str, required=True, metavar="FILE", help="Model weights .pt file")
    parser.add_argument("-mc", "--model_config", type=str, required=True, metavar="FILE", help="The tune config file with the model hyperparameters.")
    parser.add_argument("-ec", "--experiment_config", type=str, required=True, metavar="FILE", help='The experiment config used to modify the data.')
    parser.add_argument("-d", "--data", type=str, required=True, metavar="FILE", help='Input data')
    parser.add_argument("-o", "--output", type=str, required=True, metavar="FILE", help="output predictions csv file")
    parser.add_argument("--split", type=int, help="The split of the data to use (default: None)")
    parser.add_argument("--return_labels", type=bool, default=True, help="return the labels with the prediction (default: True)")

    args = parser.parse_args()
    return args

def load_model(model_class: object, weight_path: str, mconfig: dict) -> object:
    """
    Load the model with its hyperparameters and weights.
    """
    hyperparameters = mconfig["model_params"]
    model = model_class(**hyperparameters)
    model.load_state_dict(torch.load(weight_path))
    return model

def get_batch_size(mconfig: dict) -> int:
    """
    Get the batch size from the model tune config dict.

    If batch_size in model config, use it.
    Otherwise use the default of 256.

    TODO not sure how much this batch size is gonna 
    affect the computation of forward prediction.
    For the moment, just set to this.
    """
    batch_size = 256
    if "data_params" in mconfig:
        if "batch_size" in mconfig["data_params"]:
            batch_size = mconfig["data_params"]["batch_size"]
    return batch_size

def parse_y_keys(y: dict, data: pl.DataFrame, y_type='pred'):
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

def add_meta_info(data: pl.DataFrame, y: dict):
    """
    Add the meta columns to the dictionary of predictions and labels.
    In this way the output file can also contain the meta information.
    """
    # get meta keys
    keys = get_meta_keys(data.columns)

    # add meta info
    for key in keys:
        y[key] = data[key].to_list()

    return y

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

def main(model_path: str, weight_path: str, mconfig_path: str, econfig_path: str, data_path: str, output: str, return_labels: bool, split: Tuple[None, int]) -> None:
    
    # load tune output config into dictionary
    with open(mconfig_path, 'r') as in_json:
        mconfig = json.load(in_json)

    # load model
    model_class = import_class_from_file(model_path)
    model = load_model( model_class, weight_path, mconfig )

    # read experiment config and retrieve experiment name and then initialize the experiment class
    with open(econfig_path, 'r') as in_json:
        d = json.load(in_json)
        experiment_name = d["experiment"]
    initialized_experiment_class = get_experiment(experiment_name)

    # load and encode data into dataloder
    dataloader = DataLoader(
                    TorchDataset(data_path, initialized_experiment_class, split=split), 
                    batch_size=get_batch_size(mconfig), 
                    shuffle=False)

    # predict
    out = PredictWrapper(model, dataloader).predict(return_labels=return_labels)
    if return_labels:
        y_pred, y_true = out
    else:
        y_pred, y_true = out, {}

    # conver tensors to list
    # otherwise polars cannot process this dictionary properly later when converting it into data frame
    y_pred = {k: v.tolist() for k,v in y_pred.items()}
    y_true = {k: v.tolist() for k,v in y_true.items()}
    
    # make the keys coherent with the columns from input data csv
    data = pl.read_csv(data_path)
    y_pred = parse_y_keys(y_pred, data, y_type='pred')
    y_true = parse_y_keys(y_true, data, y_type='label')

    # parse predictions, labels and meta info into a data frame
    y = {**y_pred, **y_true}
    y = add_meta_info(data, y)
    df = pl.from_dict(y)

    # save output 
    df.write_csv(output)

if __name__ == "__main__":
    args = get_args()
    main(args.model, args.weight, args.model_config, args.experiment_config, args.data, args.output, args.return_labels, args.split)