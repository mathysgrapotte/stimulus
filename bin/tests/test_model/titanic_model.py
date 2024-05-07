import torch
import torch.nn as nn
from collections.abc import Callable

class ModelTitanic(nn.Module):
    def __init__(self, nb_neurons_intermediate_layer: int = 7, nb_intermediate_layers: int = 3, nb_classes: int = 2):
        super(ModelTitanic, self).__init__()
        self.input_layer = nn.Linear(7, nb_neurons_intermediate_layer)
        self.intermediate = nn.modules.ModuleList([nn.Linear(nb_neurons_intermediate_layer, nb_neurons_intermediate_layer) for _ in range(nb_intermediate_layers)])
        self.output_layer = nn.Linear(nb_neurons_intermediate_layer, nb_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pclass: torch.tensor, sex: torch.tensor, age: torch.tensor, sibsp: torch.tensor, parch: torch.tensor, fare: torch.tensor, embarked: torch.tensor):
        # print all inputs
        x = torch.stack((pclass, sex, age, sibsp, parch, fare, embarked), dim=1).float()
        x = self.relu(self.input_layer(x))
        for layer in self.intermediate:
            x = self.relu(layer(x))
        x = self.softmax(self.output_layer(x))
        return x
    
    def compute_loss(self, loss_fn, output, survived):
        return loss_fn(output, survived)
    
    def batch(self, x: dict, y: dict, loss_fn: Callable, optimizer: Callable = None):
        output = self.forward(**x)
        loss = self.compute_loss(loss_fn, output, **y)

        if optimizer is None:
            return loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss # return the main batch loss, later used for computing the validation