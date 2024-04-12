import torch 
from abc import ABC, abstractmethod

## this is a simple model that takes as input a 1D tensor of any size, apply some convolutional layer and outputs a single value using a maxpooling layer and a softmax function.

CONFIG_EXAMPLE = {

    'model_params': {
        'kernel_size': 3,
        'pool_size': 2,
        'loss_fn': {'loss_fn1': {'function': 'MSELoss', 'target': 'hola'},
                    'loss_fn2': {'function': 'CrossEntropyLoss', 'target': 'hola'}
        }
    },

    'optimizer': {
        'name': 'Adam',
        'params':{}},

    'epochs': 10,

    'lr' : 0.001,

    'data_params': {
        'batch_size': 64
    }
}

class AbstractModel(torch.nn.Module, ABC):

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self) -> torch.Tensor:
        """
        One step of the model with a forward pass and a loss calculation.
        """
        raise NotImplementedError
    
class SimpleModel(AbstractModel):
    def __init__(self, kernel_size: int = 3, pool_size: int = 2):
        super(SimpleModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=4, out_channels=1, kernel_size=kernel_size)
        self.pool = torch.nn.MaxPool1d(pool_size, pool_size)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, hello: torch.Tensor) -> torch.Tensor:
        x = hello.permute(0, 2, 1).to(torch.float32)  # permute the two last dimensions of hello 
        x = self.conv1(x)
        x = self.pool(x)
        x = self.softmax(x)
        return x
    
    def step(self, x, y, loss_dict) -> torch.Tensor:
        output = self(**x)
        loss = {}
        for key, dic in loss_fn.items():
            loss[key] = dic['function'](output, y[dic['target']].to(torch.float32))
        loss = sum(loss.values())
        return loss