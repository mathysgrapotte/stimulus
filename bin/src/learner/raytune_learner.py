import torch.nn as nn
import torch.optim as optim 
import torch
from torch.utils.data import DataLoader
from ray.tune import Trainable
from ..data.handlertorch import TorchDataset
from typing import Any, Dict
import importlib
from ray import train, tune
import ray.tune.schedulers as schedulers
import os
import json


        
class TuneTrainWrapper():
    def __init__(self, config_path, model_path, experiment_path, data_path):
        """
        Initialize the TuneWrapper with the paths to the config, model, experiment and data.
        """
        # Load the config
        self.config = {}
        with open(config_path, "r") as f:
            # TODO figure out a better way to load the config
            self.config =eval(f.read())
        
        self.config["model"] = model_path
        self.config["experiment"] = experiment_path

        try :
            assert os.path.exists(data_path)
            self.config["data_path"] = data_path
        except AssertionError:

            raise ValueError("Data path does not exist. Given path:" + data_path)           

        try:
            self.scheduler = getattr(schedulers, self.config["scheduler"]["name"])( **self.config["scheduler"]["params"])
        except AttributeError:
            raise ValueError(f"Invalid optimizer: {self.config['scheduler']['name']}, check PyTorch for documentation on available optimizers")

        # Get the tune config
        try: 
            self.tune_config = getattr(tune, self.config["tune_config"]["name"])(scheduler = self.scheduler, **self.config["tune_config"]["params"])
        except AttributeError:
            raise ValueError(f"Invalid tune_config: {self.config['tune_config']['name']}, check PyTorch for documentation on how a tune_config should be defined")

        # Get checkpoint config 
        try:
            # TODO: once the file reading is fixed, add the checkpoint_at_end parameter in the config file directly
            self.checkpoint_config = getattr(train, self.config["checkpoint_config"]["name"])(checkpoint_at_end = False, **self.config["checkpoint_config"]["params"])
        except AttributeError:
            raise ValueError(f"Invalid checkpoint_config: {self.config['checkpoint_config']['name']}, check PyTorch for documentation on how a checkpoint_config should be defined")

        # Get the run config
        try:
            self.run_config = getattr(train, self.config["run_config"]["name"])(checkpoint_config = self.checkpoint_config, **self.config["run_config"]["params"])
        except AttributeError:
            raise ValueError(f"Invalid run_config: {self.config['run_config']['name']}, check PyTorch for documentation on how a run_config should be defined")

        # substite in the checkpoint_config everything that is a bool in string with the actual bool
        self.tuner = None
        self.best_config = None
        self.results = None
        self.trainer = None

    def _prep_tuner(self):
        """
        Prepare the tuner with the configs.
        """
        self.tuner = tune.Tuner(TuneModel,
                            tune_config= self.tune_config,
                            param_space=self.config,
                            run_config=self.run_config,
                        )

    def tune(self, overwrite=False):
        """
        Run the tuning process.
        """
        if overwrite:
            self.tuner = None
            self.best_config = None
            
        if self.tuner is None:
            self._prep_tuner()
            self.results = self.tuner.fit()
            best_config = os.path.join(self.results.get_best_result().path, "params.json")

            print(f"Best config: {best_config}")
            try: 
                assert os.path.exists(best_config)
            except AssertionError:
                raise ValueError("Best config file not found.")
            with open(best_config, "r") as f:
                self.best_config = eval(f.read())
        else:
            raise ValueError("Tuner already exists - if you want to overwrite it, please set overwrite=True.")     

    def store_best_config(self, path):
        """
        Store the best config in a file.
        """
        with open(path, "w") as f:
            f.write(str(self.best_config))            
    
    def train(self, config = None): 
        """
        Train the model with the config.
        """
        if config is None:
            config = self.best_config
        try: 
            assert config is not None
        except AssertionError:
            raise ValueError("No config provided - please provide a config to train the model with or tune first by calling the tune() method.")
        
        self.trainer = TuneModel(config=config)
        for i in range(config["epochs"]):
            self.trainer.step()


class TuneModel(Trainable):

    def setup(self, config: dict):
        """
        Get the model, loss function(s), optimizer, train and test data from the config.
        """

        # Load model from string path
        module_name, class_name = config["model"].rsplit('.', 1)
        module = importlib.import_module(module_name)
        model = getattr(module, class_name)
        self.model = model(**config["model_params"])

        # Add data path
        self.data_path = config["data_path"]

        # Add experiment
        module_name, class_name = config["experiment"].rsplit('.', 1)
        module = importlib.import_module(module_name)
        self.experiment = getattr(module, class_name)()

        # Get the loss function(s) from the config model params
        # Note that the loss function(s) are stored in a dictionary, 
        # where the key is the name of the loss function and the value is the loss function itself.
        self.loss_dict = config["loss_fn"]
        for key, loss_fn in self.loss_dict.items():
            try:
                #self.loss_dict[key] = getattr(nn, loss_fn)()
                self.loss_dict[key] = loss_fn
            except AttributeError:
                raise ValueError(f"Invalid loss function: {loss_fn}, check PyTorch for documentation on available loss functions")
        
        # get the optimizer from the config
        try:
            self.optimizer = getattr(optim, config["optimizer"]["name"])(self.model.parameters(), **config["optimizer"]["params"])
        except AttributeError:
            raise ValueError(f"Invalid optimizer: {config['optimizer']['name']}, check PyTorch for documentation on available optimizers")

        # get epochs from the config
        self.epochs = config['epochs']

        # get step size from the config
        self.step_size = config['step_size']

        # get learning rate
        self.lr = config['lr']

        # get the train and validation data from the config
        # run dataloader on them
        self.training = DataLoader(TorchDataset(self.data_path, self.experiment, split=0), batch_size=config['data_params']['batch_size'])
        self.validation = DataLoader(TorchDataset(self.data_path, self.experiment, split=1), batch_size=config['data_params']['batch_size'])

    def step(self) -> dict:
        """
        Train the model for one epoch.
        For each batch in the training data, calculate the loss and update the model parameters.
        This calculation is performed based on the model's step function.
        At the end, return the objective metric(s) for the tuning process.
        """
        loss = 0.0
        self.model.train()
        for step_size in range(self.step_size):
            for x, y, meta in self.training:
                self.optimizer.zero_grad()
                current_loss = self.model.step(x, y, self.loss_dict)
                loss += current_loss.item()
                current_loss.backward()
                self.optimizer.step()
            loss /= len(self.training)
        return self.objective()

    def objective(self):
        """
        Compute the objective metric(s) for the tuning process.
        """
        return {"val_loss": self.compute_val_loss()}

    def compute_val_loss(self):
        """
        Compute loss on the validation data.
        For each batch in the validation data, calculate the loss.
        This calculation is performed based on the model's step function.
        Then retun the average loss.
        """
        loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for x, y, meta in self.validation:
                loss += self.model.step(x, y, self.loss_dict).item()
        loss /= len(self.validation)
        return loss
        
    def export_model(self, export_dir: str) -> None:
        torch.save(self.model.state_dict(), export_dir)

    def load_checkpoint(self, checkpoint: Dict | None):
        if checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_checkpoint(self, checkpoint_dir: str) -> Dict | None:
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_dir)
        return checkpoint
        



        


