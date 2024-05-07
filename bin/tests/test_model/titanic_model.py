import torch
import torch.nn as nn
from typing import Callable, Dict, Tuple, Optional

class ModelTitanic(nn.Module):
    def __init__(self, nb_neurons_intermediate_layer: int = 7, nb_intermediate_layers: int = 3, nb_classes: int = 2):
        super(ModelTitanic, self).__init__()
        self.input_layer = nn.Linear(7, nb_neurons_intermediate_layer)
        self.intermediate = nn.modules.ModuleList([nn.Linear(nb_neurons_intermediate_layer, nb_neurons_intermediate_layer) for _ in range(nb_intermediate_layers)])
        self.output_layer = nn.Linear(nb_neurons_intermediate_layer, nb_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pclass: torch.Tensor, sex: torch.Tensor, age: torch.Tensor, sibsp: torch.Tensor, parch: torch.Tensor, fare: torch.Tensor, embarked: torch.Tensor):
        x = torch.stack((pclass, sex, age, sibsp, parch, fare, embarked), dim=1).float()
        x = self.relu(self.input_layer(x))
        for layer in self.intermediate:
            x = self.relu(layer(x))
        x = self.softmax(self.output_layer(x))
        return {'survived':x}
    
    def compute_loss(self, loss_fn, output, survived):
        return loss_fn(output, survived)
    
    def batch(self, x: dict, y: dict, loss_fn: Callable, optimizer: Optional[Callable] = None) -> Tuple[float, dict]:
        output = self.forward(**x)
        loss = self.compute_loss(loss_fn, output['survived'], y['survived'])
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        output = {k:torch.argmax(v, dim=1) for k,v in output.items()}   # this solves the issue of different shape between output and labels. TODO needs to make it more general so that classification outputs can also be evaluated by Performance
        return loss, output
