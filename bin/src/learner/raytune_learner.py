import torch.nn as nn
import torch.optim as optim 
import torch
from torch.utils.data import DataLoader
from ray.tune import Trainable
from ..data.handlertorch import handlertorch as handlertorch
from typing import Any, Dict

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

class TuneModel(Trainable):
    def setup(self, config: dict, model: nn.Module, data_path: str, experiment: Any):
        self.model = model(**config["model_params"])

        try:
            self.loss_fn = getattr(nn, config["loss_fn"])()
        except AttributeError:
            raise ValueError(f"Invalid loss function: {config['loss_fn']}, check PyTorch for documentation on available loss functions")
        
        try:
            self.optimizer = getattr(optim, config["optimizer"]["name"])(self.model.parameters(), **config["optimizer"]["params"])
        except AttributeError:
            raise ValueError(f"Invalid optimizer: {config['optimizer']['name']}, check PyTorch for documentation on available optimizers")
        
        self.train = DataLoader(handlertorch.TorchDataset(data_path, experiment, split=0), batch_size=config['data_params']['batch_size'])
        self.test = DataLoader(handlertorch.TorchDataset(data_path, experiment, split=1), batch_size=config['data_params']['batch_size'])


    def compute_val_loss(self):
        self.model.eval()
        with torch.no_grad():
            for x, y, _, _, _ in self.test:
                output = self.model(x)
                loss = self.loss_fn(output, y)
        return loss.item()

    def objective(self):
        return {"val_loss": self.compute_val_loss()}

    def step(self):
        self.model.train()
        for x, y, mask_x, mask_y, meta in self.train:
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss_fn(output, y)
            # if mask_x is not None, apply it to the loss
            if mask_x is not None: #TODO this is most likely wrong, check loss dimensions to solve this issue. 
                loss = loss * mask_x
            loss.backward()
            self.optimizer.step()

        return self.objective()
    
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
        

        


        


