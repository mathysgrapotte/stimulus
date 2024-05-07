import os
import ray.tune.schedulers as schedulers
import torch
import torch.nn as nn
import torch.optim as optim 
from ray import train, tune
from ray.tune import Trainable
from torch.utils.data import DataLoader

from ..data.handlertorch import TorchDataset
from ..utils.yaml_model_schema import YamlRayConfigLoader
from ..utils.performance import Performance


class TuneWrapper():
    def __init__(self, config_path: str, model_class: nn.Module, data_path: str, experiment_object: object) -> None:
        """
        Initialize the TuneWrapper with the paths to the config, model, and data.
        """
        self.config = YamlRayConfigLoader(config_path).get_config()
        self.config["model"] = model_class
        self.config["experiment"] = experiment_object

        if not os.path.exists(data_path):
            raise ValueError("Data path does not exist. Given path:" + data_path)
        self.config["data_path"] = os.path.abspath(data_path)
        
        # build the tune config
        self.config["tune"]["tune_params"]["scheduler"] = getattr(schedulers, self.config["tune"]["scheduler"]["name"])( **self.config["tune"]["scheduler"]["params"])
        self.tune_config = tune.TuneConfig(**self.config["tune"]["tune_params"])

        # build the run config
        self.checkpoint_config = train.CheckpointConfig(checkpoint_at_end=True) #TODO implement checkpoiting
        self.run_config = train.RunConfig(checkpoint_config=self.checkpoint_config) #TODO implement run_config (in tune/run_params for the yaml file)
        
        self.tuner = self.tuner_initialization()

    def tuner_initialization(self) -> tune.Tuner:
        """
        Prepare the tuner with the configs.
        """
        return tune.Tuner(TuneModel,
                            tune_config= self.tune_config,
                            param_space=self.config,
                            run_config=self.run_config,
                        )

    def tune(self) -> None:
        """
        Run the tuning process.
        """
        return self.tuner.fit() 

class TuneModel(Trainable):

    def setup(self, config: dict) -> None:
        """
        Get the model, loss function(s), optimizer, train and test data from the config.
        """

        # Initialize model with the config params
        self.model = config["model"](**config["model_params"])

        # Add data path
        self.data_path = config["data_path"]

        # Use the already initialized experiment class      
        self.experiment = config["experiment"]

        # Get the loss function(s) from the config model params
        # Note that the loss function(s) are stored in a dictionary, 
        # where the key is the name of the loss function and the value is the loss function itself.
        self.loss_dict = config["loss_params"]
        for key, loss_fn in self.loss_dict.items():
            try:
                self.loss_dict[key] = getattr(nn, loss_fn)()
            except AttributeError:
                raise ValueError(f"Invalid loss function: {loss_fn}, check PyTorch for documentation on available loss functions")
        
        # get the optimizer parameters
        optimizer_lr = config["optimizer_params"]["lr"]

        # get the optimizer from PyTorch
        self.optimizer = getattr(optim, config["optimizer_params"]["method"])(self.model.parameters(), lr=optimizer_lr)

        # get step size from the config
        self.step_size = config["tune"]['step_size']

        # get the train and validation data from the config
        # run dataloader on them
        self.training = DataLoader(TorchDataset(self.data_path, self.experiment, split=0), batch_size=config['data_params']['batch_size'])
        self.validation = DataLoader(TorchDataset(self.data_path, self.experiment, split=1), batch_size=config['data_params']['batch_size'])

    def step(self) -> dict:
        """
        For each batch in the training data, calculate the loss and update the model parameters.
        This calculation is performed based on the model's batch function.
        At the end, return the objective metric(s) for the tuning process.
        """
        for step_size in range(self.step_size):
            for x, y, meta in self.training:
                self.model.batch(x=x, y=y, optimizer=self.optimizer, **self.loss_dict)
        return self.objective()

    def objective(self) -> dict:
        """
        Compute the objective metric(s) for the tuning process.
        """
        metrics = ['loss', 'rocauc', 'prauc', 'mcc', 'f1score', 'precision', 'recall', 'spearmanr']  # TODO maybe we report only a subset of metrics, given certain criteria (eg. if classification or regression)
        return {**{'val_'+metric : self.compute_metric(self.validation, metric) for metric in metrics},
                **{'train_'+metric : self.compute_metric(self.training, metric) for metric in metrics}}

    def compute_metric(self, data, metric: str = 'loss') -> float:
        """
        Compute the performance metric.
        Basically, it runs a foward pass on the model and returns the performance metric.
        """
        self.model.eval()
        loss = 0.0
        labels = {k:torch.tensor([]) for k in list(data)[0][1].keys()}
        predictions = {k:torch.tensor([]) for k in list(data)[0][1].keys()}   # list(data)[0] is the first batch of data, then the element [1] is y.
        with torch.no_grad():

            # for each batch, get the loss and the predictions
            for x, y, meta in data:
                current_loss, current_predictions = self.model.batch(x, y, **self.loss_dict)

                # update values
                loss += current_loss.item()
                for k in current_predictions.keys():
                    labels[k] = torch.cat((labels[k], y[k]))
                    predictions[k] = torch.cat((predictions[k], current_predictions[k]))

        # return metric
        if metric == 'loss':
            return loss / len(data)
        else:
            return sum(Performance(labels=labels[k], predictions=predictions[k], metric=metric).val for k in labels.keys()) / len(labels.keys())  # TODO currently we computes the average performance metric, but maybe in the future we want something different
        
    def export_model(self, export_dir: str) -> None:
        torch.save(self.model.state_dict(), export_dir)

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))
        self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))

    def save_checkpoint(self, checkpoint_dir: str) -> dict | None:
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        return checkpoint_dir
    