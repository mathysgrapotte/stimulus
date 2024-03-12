import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
from ray.tune import Trainable
from ..data.handlertorch import handlertorch as handlertorch
from typing import Any

CONFIG_EXAMPLE = {
    'model_params': {
        'kernel_size': 3,
        'pool_size': 2
    },

    'optimizer': {
        'name': 'Adam',
        'params':{}},

    'loss_fn': 'MSELoss',

    'data_params': {
        'batch_size': 64
    }
}

class TuneSimpleModel(Trainable):
    def setup(self, config: dict, model: Any, data_path: str, experiment: Any):
        self.model = model(**config["model_params"])

        try:
            self.loss_fn = getattr(nn, config["loss_fn"])()
        except AttributeError:
            raise ValueError(f"Invalid loss function: {config['loss_fn']}, check PyTorch for documentation on available loss functions and their names")
        
        try:
            self.optimizer = getattr(optim, config["optimizer"]["name"])(self.model.parameters(), **config["optimizer"]["params"])
        except AttributeError:
            raise ValueError(f"Invalid optimizer: {config['optimizer']['name']}, check PyTorch for documentation on available optimizers and their names")
        
        self.data = DataLoader(handlertorch.TorchDataset(data_path, experiment), batch_size=config['data_params']['batch_size'])

        


        


