import torch 

## this is a simple model that takes as input a 1D tensor of any size, apply some convolutional layer and outputs a single value using a maxpooling layer and a softmax function.

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 1, 3)
        self.pool = torch.nn.MaxPool1d(2, 2)
        self.fc1 = torch.nn.Linear(1, 1)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x['dna']
        x = self.conv1(x)
        x = self.pool(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x