import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, average_precision_score
from typing import Any

class Performance():
    """
    Returns the value of a given metric

    Parameters
    ----------
    labels (np.array) : labels
    predictions (np.array) : predictions
    metric (str) : the metric to compute

    Returns
    -------
    value (float) : the value of the metric

    TODO we can add more metrics here
    TODO we can adjust the threshold for positive class for some metrics
    """
    def __init__(self, labels: Any, predictions: Any, metric: str = "rocauc") -> float:
        labels = self.data2array(labels)
        predictions = self.data2array(predictions)
        if labels.shape != predictions.shape:
            raise ValueError(f"The labels have shape {labels.shape} whereas predictions have shape {predictions.shape}.")
        function = getattr(self, metric)
        self.val = function(labels, predictions)

    def data2array(self, data: Any) -> np.array:
        if isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.numpy()
        elif isinstance(data, (int, float)):
            return np.array([data])
        else:
            raise ValueError(f"The data must be a list, np.array, torch.Tensor, int or float. Instead it is {type(data)}")
    
    def rocauc(self, labels: np.array, predictions: np.array) -> float:
        return roc_auc_score(labels, predictions)

    def prauc(self, labels: np.array, predictions: np.array) -> float:
        return average_precision_score(labels, predictions)
    
    def mcc(self, labels: np.array, predictions: np.array) -> float:
        predictions = np.array([1 if p > 0.5 else 0 for p in predictions])
        return matthews_corrcoef(labels, predictions)

    def f1score(self, labels: np.array, predictions: np.array) -> float:
        predictions = np.array([1 if p > 0.5 else 0 for p in predictions])
        return f1_score(labels, predictions)

    def precision(self, labels: np.array, predictions: np.array) -> float:
        predictions = np.array([1 if p > 0.5 else 0 for p in predictions])
        return precision_score(labels, predictions)

    def recall(self, labels: np.array, predictions: np.array) -> float:
        predictions = np.array([1 if p > 0.5 else 0 for p in predictions])
        return recall_score(labels, predictions)

    def spearmanr(self, labels: np.array, predictions: np.array) -> float:
        return spearmanr(labels, predictions)[0]

