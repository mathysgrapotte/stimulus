import torch
import torch.nn as nn

class ModelTitanic(nn.Module):
    def __init__(self, nb_neurons_intermediate_layer: int = 7, nb_intermediate_layers: int = 3, nb_classes: int = 2):
        super(ModelTitanic, self).__init__()
        self.input_layer = nn.Linear(7, nb_neurons_intermediate_layer)
        self.intermediate = nn.modules.ModuleList([nn.Linear(nb_neurons_intermediate_layer, nb_neurons_intermediate_layer) for _ in range(nb_intermediate_layers)])
        self.output_layer = nn.Linear(nb_neurons_intermediate_layer, nb_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pclass, sex, age, sibsp, parch, fare, embarked):
        # print all inputs
        x = torch.stack((pclass, sex, age, sibsp, parch, fare, embarked), dim=1).float()
        x = self.relu(self.input_layer(x))
        for layer in self.intermediate:
            x = self.relu(layer(x))
        x = self.softmax(self.output_layer(x))
        return x
    
    def compute_loss(self, loss_fn, output, survived):
        return loss_fn(output, survived)
    
    def epoch(self, x, y, loss_fn, optimizer):
        # concatenate all input tensors into one tensor that should be shape (batch_size, 7)

        output = self.forward(**x)
        loss = self.compute_loss(loss_fn, output, **y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss