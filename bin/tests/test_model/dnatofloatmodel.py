import torch 
from abc import ABC, abstractmethod
import torch.nn as nn

## this is a simple model that takes as input a 1D tensor of any size, apply some convolutional layer and outputs a single value using a maxpooling layer and a softmax function.

class AbstractModel(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def batch(self) -> torch.Tensor:
        """
        One step of the model with a forward pass and a loss calculation.
        """
        raise NotImplementedError
    
class ModelSimple(AbstractModel):
    def __init__(self, kernel_size: int = 3, pool_size: int = 2):
        super(ModelSimple, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=4, out_channels=1, kernel_size=kernel_size)
        self.pool = torch.nn.MaxPool1d(pool_size, pool_size)
        self.softmax = torch.nn.Softmax(dim=1)
        self.linear = torch.nn.Linear(7,1)
    
    def forward(self, hello: torch.Tensor) -> torch.Tensor:
        x = hello.permute(0, 2, 1).to(torch.float32)  # permute the two last dimensions of hello 
        x = self.conv1(x)
        x = self.pool(x)
        x = self.softmax(x)
        x = self.linear(x)
        x = x.squeeze()
        return x
    
    def compute_loss(self, output, loss_fn, hola):
        return loss_fn(output, hola.to(torch.float32))
    
    def batch(self, x, y, loss_fn1, loss_fn2, optimizer=None) -> torch.Tensor:
        output = self(**x)
        loss1 = self.compute_loss(output, loss_fn1, **y)
        loss2 = self.compute_loss(output, loss_fn2, **y)
        if optimizer is None: # if no optimizer is passed, return the losses
            return loss1
    
        optimizer.zero_grad()
        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        optimizer.step()

        return loss1