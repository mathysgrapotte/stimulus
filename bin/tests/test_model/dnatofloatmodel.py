import torch 

## this is a simple model that takes as input a 1D tensor of any size, apply some convolutional layer and outputs a single value using a maxpooling layer and a softmax function.

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

class SimpleModel(torch.nn.Module):
    def __init__(self, kernel_size: int = 3, pool_size: int = 2):
        super(SimpleModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=4, out_channels=1, kernel_size=kernel_size)
        self.pool = torch.nn.MaxPool1d(pool_size, pool_size)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, hello: torch.Tensor) -> torch.Tensor:
        # permute the two last dimensions of hello 
        x = hello.permute(0, 2, 1).to(torch.float32)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.softmax(x)
        return x