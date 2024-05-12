#!/usr/bin/env python3

import argparse
import json
import os
import pandas as pd
import torch

from launch_utils import import_class_from_file, get_experiment
from src.analysis.analysis_default import AnalysisPerformanceTune, AnalysisRobustness

def get_args():
    
    "get the arguments when using from the commandline"
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", "--model", type=str, required=True, metavar="FILE", help='The model .py file')
    parser.add_argument("-w", "--weight", type=str, required=True, nargs="+", metavar="FILE", help="Model weights .pt file")
    parser.add_argument("-me", "--metrics", type=str, required=True, nargs="+", metavar="FILE", help='The file path for the metrics file obtained during tuning')
    parser.add_argument("-ec", "--experiment_config", type=str, required=True, nargs="+", metavar="FILE", help='The experiment config used to modify the data.')
    parser.add_argument("-mc", "--model_config", type=str, required=True, nargs="+", metavar="FILE", help="The tune config file.")
    parser.add_argument("-d", "--data", type=str, required=True, nargs="+", metavar="FILE", help='List of data files to be used for the analysis.')
    # parser.add_argument("-o", "--output", type=str, required=True, metavar="FILE", help="output report")
    parser.add_argument("-o", "--outdir", type=str, required=True, help="output directory")

    args = parser.parse_args()
    return args

def main(model_path: str, weight_list: list, mconfig_list: list, metrics_list: list, econfig_list: list, data_list: list, outdir: str):

    metrics = ["rocauc", "prauc", "mcc", "f1score", "precision", "recall"]

    # plot the performance during tuning/training
    run_analysis_performance_tune(metrics_list, metrics+["loss"], outdir)

    # check the performance of each model on each dataset test
    run_analysis_performance_model(metrics, model_path, weight_list, mconfig_list, econfig_list, data_list, outdir)

    # TODO compile all the information in one report (pdf maybe)


def run_analysis_performance_tune(metrics_list: list, metrics: list, outdir: str):
    """
    each model has a metrics file obtained during tuning/training,
    check the performance there and plot it.
    This is to track the model performance per training iteration.
    """
    for metrics_path in metrics_list:
        AnalysisPerformanceTune(metrics_path).plot_metric_vs_iteration(
            metrics=metrics, 
            output=os.path.join(outdir, metrics_path.replace("-metrics.csv", "") + "-metric_vs_iteration.png")
        )

def run_analysis_performance_model(metrics: list, model_path: list, weight_list: list, mconfig_list: list, econfig_list: list, data_list: list, outdir: str):
    """
    check the performance of each model on each dataset test
    """
    # load all the models
    model_names = []
    model_list  = []
    model_class = import_class_from_file(model_path)
    for weight_path, mconfig_path in zip(weight_list, mconfig_list):
        model = load_model(model_class, weight_path, mconfig_path)
        model_names.append( mconfig_path.split("/")[-1].replace("-config.json", "") )
        model_list.append(model)

    # read experiment config and retrieve experiment name and then initialize the experiment class
    experiment_name = None
    with open(econfig_list[0], 'r') as in_json:
        d = json.load(in_json)
        experiment_name = d["experiment"]
    initialized_experiment_class = get_experiment(experiment_name)

    # initialize analysis
    analysis = AnalysisRobustness(metrics, initialized_experiment_class, batch_size=10)

    # get the performance table of each model with each dataset
    df = analysis.get_performance_table(model_names, model_list, data_list)
    df.to_csv(os.path.join(outdir, "performance_table.csv"), index=False)

    # plot delta performance

    # get robustness score

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

if __name__ == "__main__":
    args = get_args()
    main(args.model, args.weight, args.model_config, args.metrics, args.experiment_config, args.data, args.outdir)