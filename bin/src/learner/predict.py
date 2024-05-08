import torch
from torch.utils.data import DataLoader
from ..data.handlertorch import TorchDataset
from ..utils.performance import Performance

class PredictWrapper():
    """
    A wrapper to predict the output of a model on a dataset.
    It also provides the functionalities to measure the performance of the model.
    """
    def __init__(self, model: object, data_path: str, experiment: object, loss_dict: dict, split: int, batch_size: int):
        self.model = model
        self.loss_dict = loss_dict
        self.dataloader = DataLoader(TorchDataset(data_path, experiment, split=split), batch_size=batch_size, shuffle=False)

    def predict(self) -> dict:
        """
        Get the model predictions.

        Basically, it runs a foward pass on the model for each batch, 
        gets the predictions and concatenate them for all batches.
        Since the returned `current_predictions` are formed by tensors computed for one batch,
        the final `predictions` are obtained by concatenating them.

        At the end it returns `predictions` as a dictionary of tensors with the same keys as `y`.
        """
        self.model.eval()
        predictions = {k:[] for k in list(self.dataloader)[0][1].keys()}

        # get the predictions for each batch
        with torch.no_grad():
            for x, y, meta in self.dataloader:
                current_predictions = self.model.batch(x, y, **self.loss_dict)[1]
                for k in current_predictions.keys():
                    predictions[k].append(current_predictions[k])

        # return the predictions as a dictionary of tensors for the entire dataset
        return {k: torch.cat(v) for k, v in predictions.items()}

    def get_labels(self) -> dict:
        """
        Returns the labels of the data.

        It also gets the labels for each batch, and then concatenate them all together.
        At the end it returns `labels` as a dictionary of tensors with the same keys as `y`.
        """
        labels = {k:[] for k in list(self.dataloader)[0][1].keys()}
        for _, y, _ in self.dataloader:
            for k in y.keys():
                labels[k].append(y[k])
        return {k: torch.cat(v) for k, v in labels.items()}

    def compute_metric(self, metric: str = 'loss') -> float:
        """
        Wrapper to compute the performance metric.
        """
        if metric == 'loss':
            return self.compute_loss()
        else:
            return self.compute_other_metric(metric)
        
    def compute_loss(self) -> float:
        """
        Compute the loss.

        The current implmentation basically computes the loss for each batch and then averages them.
        TODO we could potentially summarize the los across batches in a different way. 
        Or sometimes we may potentially even have 1+ losses.
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
        return sum(Performance(labels=labels[k], predictions=predictions[k], metric=metric).val for k in labels.keys()) / len(labels.keys())