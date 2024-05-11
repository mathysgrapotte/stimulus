#!/usr/bin/env python3

import argparse
import json
import os
import torch

from launch_utils import import_class_from_file, get_experiment
from src.analysis.analysis_default import AnalysisPerformanceTune, AnalysisPerformanceModel

def get_args():
    
    "get the arguments when using from the commandline"
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", "--model", type=str, required=True, metavar="FILE", help='The model .py file')
    parser.add_argument("-w", "--weight", type=str, required=True, metavar="FILE", help="Model weights .pt file")
    parser.add_argument("-me", "--metrics", type=str, required=True, metavar="FILE", help='The file path for the metrics file obtained during tuning')
    parser.add_argument("-e", "--experiment_config", type=str, required=True, metavar="FILE", help='The experiment config used to modify the data.')
    parser.add_argument("-t", "--tune_config", type=str, required=True, metavar="FILE", help="The tune config file.")
    parser.add_argument("-d", "--data", type=str, required=True, nargs="+", metavar="FILE", help='List of data files to be used for the analysis.')
    # parser.add_argument("-o", "--output", type=str, required=True, metavar="FILE", help="output report")
    parser.add_argument("-o", "--outdir", type=str, required=True, help="output directory")

    args = parser.parse_args()
    return args

def main(model_path: str, weight_path: str, tune_config: str, metrics_path: str, experiment_config: str, data_list: list, outdir: str):

    # load model
    with open(tune_config, 'r') as in_json:
        model_config = json.load(in_json)["model_params"]
    model = import_class_from_file(model_path)(**model_config)
    model.load_state_dict(torch.load(weight_path))

    # read experiment config and retrieve experiment name and then initialize the experiment class
    experiment_name = None
    with open(experiment_config, 'r') as in_json:
        d = json.load(in_json)
        experiment_name = d["experiment"]
    initialized_experiment_class = get_experiment(experiment_name)

    # save plot: metric vs training iteration
    metrics = ["rocauc", "prauc", "mcc", "f1score", "precision", "recall"]
    AnalysisPerformanceTune(metrics_path).plot_metric_vs_iteration(
        metrics=metrics+["loss"], 
        output=os.path.join(outdir, "metric_vs_iteration.png"))

    # check model performance
    for data_path in data_list:
        outfile = os.path.join(outdir, data_path.split("/")[-1].split(".")[0] + "_metric.csv")
        AnalysisPerformanceModel(model, data_path, initialized_experiment_class, batch_size=10).get_performance_table(metrics=metrics, output=outfile)

    # TODO compile all the information in one report (pdf maybe)


if __name__ == "__main__":
    args = get_args()
    main(args.model, args.weight, args.tune_config, args.metrics, args.experiment_config, args.data, args.outdir)