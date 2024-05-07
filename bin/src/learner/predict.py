import torch
from torch.utils.data import DataLoader
from ..data.handlertorch import TorchDataset
from ..utils.performance import Performance

class PredictWrapper():
    def __init__(self, model, data_path: str, experiment: object, loss_dict: dict, split: int, batch_size: int):
        self.model = model
        self.loss_dict = loss_dict
        self.dataloader = DataLoader(TorchDataset(data_path, experiment, split=split), batch_size=batch_size, shuffle=False)

    def predict(self) -> dict:
        """
        Predicts the output of the model on the data.
        """
        self.model.eval()
        predictions = {k:[] for k in list(self.dataloader)[0][1].keys()}   # list(self.dataloader)[0] is the first batch of dataloader, then the element [1] is y.
        with torch.no_grad():
            for x, y, meta in self.dataloader:
                current_predictions = self.model.batch(x, y, **self.loss_dict)[1]
                for k in current_predictions.keys():
                    predictions[k].append(current_predictions[k])
        return {k: torch.cat(v) for k, v in predictions.items()}

    def get_labels(self) -> dict:
        """
        Returns the labels of the data.
        """
        labels = {k:[] for k in list(self.dataloader)[0][1].keys()}
        for _, y, _ in self.dataloader:
            for k in y.keys():
                labels[k].append(y[k])
        return {k: torch.cat(v) for k, v in labels.items()}

    def compute_metric(self, metric: str = 'loss') -> float:
        """
        Compute the performance metric.
        Basically, it runs a foward pass on the model and returns the performance metric.
        """
        if metric == 'loss':
            return self.compute_loss()
        else:
            return self.compute_other_metric(metric)
        
    def compute_loss(self) -> float:
        """
        Compute the loss.
        """
        self.model.eval()
        loss = 0.0
        with torch.no_grad():
            for x, y, meta in self.dataloader:
                current_loss = self.model.batch(x, y, **self.loss_dict)[0]
                loss += current_loss.item()
        return loss / len(self.dataloader)

    def compute_other_metric(self, metric: str) -> float:
        """
        Compute the performance metric.

        # TODO currently we computes the average performance metric across target y, but maybe in the future we want something different
        """
        self.model.eval()
        labels = self.get_labels()
        predictions = self.predict()
        val = sum(Performance(labels=labels[k], predictions=predictions[k], metric=metric).val for k in labels.keys()) / len(labels.keys())
        return val