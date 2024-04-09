import torch.nn as nn
import torch.optim as optim 
import torch
from torch.utils.data import DataLoader
from ray.tune import Trainable
from ..data.handlertorch import TorchDataset
from typing import Any, Dict

#TODO write a wrapper to the TuneModel class that takes a model, a experiment, a data_path and a config, add the model, data_path and experiment to the config and pass it to the TuneModel class.

class TuneModel(Trainable):

    def setup(self, config: dict):
        """
        Get the model, loss function(s), optimizer, train and test data from the config.
        """
        # get the model from the config
        self.model = config['model'](**config["model_params"])

        # Get the loss function(s) from the config model params
        # Note that the loss function(s) are stored in a dictionary, 
        # where the key is the name of the loss function and the value is the loss function itself.
        self.loss_dict = config["model_params"]["loss_fn"]
        for key, loss_fn in self.loss_dict.items():
            try:
                self.loss_dict[key] = getattr(nn, loss_fn)()
            except AttributeError:
                raise ValueError(f"Invalid loss function: {loss_fn}, check PyTorch for documentation on available loss functions")
        
        # get the optimizer from the config
        try:
            self.optimizer = getattr(optim, config["optimizer"]["name"])(self.model.parameters(), **config["optimizer"]["params"])
        except AttributeError:
            raise ValueError(f"Invalid optimizer: {config['optimizer']['name']}, check PyTorch for documentation on available optimizers")

        # get epochs from the config
        self.epochs = config['epochs']

        # get the train and validation data from the config
        # run dataloader on them
        self.train = DataLoader(TorchDataset(config['data_path'], config['experiment'], split=0), batch_size=config['data_params']['batch_size'])
        self.validation = DataLoader(TorchDataset(config['data_path'], config['experiment'], split=1), batch_size=config['data_params']['batch_size'])

    def run_epoch(self):
        """
        Train the model for one epoch.
        For each batch in the training data, calculate the loss and update the model parameters.
        This calculation is performed based on the model's step function.
        """
        loss = 0.0
        self.model.train()
        for x, y, meta in self.train:
            self.optimizer.zero_grad()
            current_loss = self.model.step(x, y, self.loss_dict)
            loss += current_loss.item()
            current_loss.backward()
            self.optimizer.step()
        loss /= len(self.train)
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
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        

        


        


