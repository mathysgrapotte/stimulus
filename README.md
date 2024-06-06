Stochastic Testing and Input Manipulation for Unbiased Learning Systems (STIMULUS) is an end-to-end nextflow based pipeline for statistically testing training procedures of machine learning models. 

Deep learning model development in natural science is an empirical and costly process. Users must define a pre-processing pipeline, an architecture, find the best parameters for said architecture and iterate over this process.

Leveraging the power of Nextflow (polyglotism, container integration, scalable on the cloud), we propose STIMULUS, an open-source software built to automatize deep learning model development for genomics.

STIMULUS takes as input a user defined PyTorch model, a dataset, a configuration file to describe the pre-processing steps to be performed, and a range of parameters for the PyTorch model.  It then transforms the data according to all possible pre-processing steps, finds the best architecture parameters for each of the transformed datasets, performs sanity checks on the models and train a minimal deep learning version for each dataset/architecture.

Those experiments are then compiled into an intuitive report, making it easier for scientists to pick the best design choice to be sent to large scale training.

## Code requirements

### Data



### Model 

1. The model class you want to train should start with "Model"
```python

import torch
import torch.nn as nn

class SubClass(nn.Module):
    """
    a subclass
    """

class ModelClass(nn.Module):
    """
    the PyTorch model to be trained by Stimulus, can use SubClass if needed
    """

```

2. The model "forward" function should