import torch
from ..utils.performance import Performance
from ..utils.generic_utils import ensure_at_least_1d

class PredictWrapper():
    """
    A wrapper to predict the output of a model on a datset loaded into a torch DataLoader.
    It also provides the functionalities to measure the performance of the model.
    """
    def __init__(self, model: object, dataloader: object, loss_dict: dict = None):
        self.model = model
        self.dataloader = dataloader
        self.loss_dict = loss_dict
        try:
            self.model.eval()
        except:
            print("warning: not able to run model.eval")

    def predict(self) -> dict:
        """
        Get the model predictions.

        Basically, it runs a foward pass on the model for each batch, 
        gets the predictions and concatenate them for all batches.
        Since the returned `current_predictions` are formed by tensors computed for one batch,
        the final `predictions` are obtained by concatenating them.

        At the end it returns `predictions` as a dictionary of tensors with the same keys as `y`.
        """
        predictions = {k:[] for k in list(self.dataloader)[0][1].keys()}
        # get the predictions for each batch
        with torch.no_grad():
            for x, y, _ in self.dataloader:
                current_predictions = self.model(**x)
                current_predictions = self.handle_predictions(current_predictions, y)
                for k in current_predictions.keys():
                    # it might happen that the batch consists of one element only so the torch.cat will fail. To prevent this the function to ensure at least one dimensionality is called.
                    predictions[k].append(ensure_at_least_1d(current_predictions[k]))
        # return the predictions as a dictionary of tensors for the entire dataset.
        return {k: torch.cat(v) for k, v in predictions.items()}

    def handle_predictions(self, predictions, y) -> dict:
        """
        Handle the model outputs from forward pass.
        """
        if len(y) == 1:
            return {list(y.keys())[0]: predictions}
        else:
            return {k:v for k, v in zip(y.keys(), predictions)}

    def get_labels(self) -> dict:
        """
        Returns the labels of the data.

        It also gets the labels for each batch, and then concatenate them all together.
        At the end it returns `labels` as a dictionary of tensors with the same keys as `y`.
        """
        labels = {k:[] for k in list(self.dataloader)[0][1].keys()}
        for _, y, _ in self.dataloader:
            for k in y.keys():
                # it might happen that the batch consists of one element only so the torch.cat will fail. To prevent this the function to ensure at least one dimensionality is called.
                labels[k].append(ensure_at_least_1d(y[k]))
        return {k: torch.cat(v) for k, v in labels.items()}
    
    def compute_metrics(self, metrics: list) -> dict:
        """
        Wrapper to compute the performance metrics.
        """
        return {m: self.compute_metric(m) for m in metrics}

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
        if self.loss_dict is None:
            raise ValueError("Loss function is not provided.")
        loss = 0.0
        with torch.no_grad():
            for x, y, meta in self.dataloader:
                # the loss_dict could be unpacked with ** and the function declaration handle it differently like **kwargs. to be decided, personally find this more clean and understable.
                current_loss = self.model.batch(x=x, y=y, **self.loss_dict)[0]
                loss += current_loss.item()
        return loss / len(self.dataloader)

    def compute_other_metric(self, metric: str) -> float:
        """
        Compute the performance metric.

        # TODO currently we computes the average performance metric across target y, but maybe in the future we want something different
        """
        # if labels not in self
        if not hasattr(self, 'labels'):
            self.labels = self.get_labels()
        if not hasattr(self, 'predictions'):
            self.predictions = self.predict()
        return sum(Performance(labels=self.labels[k], predictions=self.predictions[k], metric=metric).val for k in self.labels.keys()) / len(self.labels.keys())