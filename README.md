# STIMULUS

Stochastic Testing and Input Manipulation for Unbiased Learning Systems (STIMULUS) is an end-to-end nextflow based pipeline for statistically testing training procedures of machine learning models. 

Deep learning model development in natural science is an empirical and costly process. Users must define a pre-processing pipeline, an architecture, find the best parameters for said architecture and iterate over this process.

Leveraging the power of Nextflow (polyglotism, container integration, scalable on the cloud), we propose STIMULUS, an open-source software built to automatize deep learning model development for genomics.

STIMULUS takes as input a user defined PyTorch model, a dataset, a configuration file to describe the pre-processing steps to be performed, and a range of parameters for the PyTorch model.  It then transforms the data according to all possible pre-processing steps, finds the best architecture parameters for each of the transformed datasets, performs sanity checks on the models and train a minimal deep learning version for each dataset/architecture.

Those experiments are then compiled into an intuitive report, making it easier for scientists to pick the best design choice to be sent to large scale training.

![alt text](https://github.com/mathysgrapotte/stimulus/blob/readme/visual_assets/stimulus_overview.png)

# Quick start

## Code requirements

### Data

The data is provided as a csv where the header columns are in the following format : name:type:class

*name* is user given (note that it has an impact on experiment definition).

*type* is either "input", "meta", or "label". "input" types are fed into the mode, "meta" types are registered but not transformed nor fed into the models and "label" is used as a training label. 

*class* is a supported class of data for which encoding methods have been created, please raise an issue on github or contribute a PR if a class of your interest is not implemented

#### csv general example

| input1:input:input_type | input2:input:input_type | meta1:meta:meta_type | label1:label:label_type | label2:label:label_type |
|-------------------------|-------------------------|----------------------|-------------------------|-------------------------|
| sample1 input1          | sample1 input2          | sample1 meta1        | sample1 label1          | sample1 label2          |
| sample2 input1          | sample2 input2          | sample2 meta1        | sample2 label1          | sample2 label2          |
| sample3 input1          | sample3 input2          | sample3 meta1        | sample3 label1          | sample3 label2          |


#### csv specific example


| mouse_dna:input:dna     | mouse_rnaseq:label:float|
|-------------------------|-------------------------|
| ACTAGGCATGCTAGTCG       | 0.53                    |
| ACTGGGGCTAGTCGAA        | 0.23                    |
| GATGTTCTGATGCT          | 0.98                    |

### Model 

In STIMULUS, users input a .py file containing a model written in pytorch (see examples in bin/tests/models)

Said models should obey to minor standards:

1. The model class you want to train should start with "Model", there should be exactly one class starting with "Model".
```python

import torch
import torch.nn as nn

class SubClass(nn.Module):
    """
    a subclass, this will be invisible to Stimulus
    """

class ModelClass(nn.Module):
    """
    the PyTorch model to be trained by Stimulus, can use SubClass if needed
    """

class ModelAnotherClass(nn.Module):
    """
    uh oh, this will return an error as there are two classes starting with Model
    """

```

2. The model "forward" function should have input variables with the **same names** as the defined input names in the csv input file

```python

import torch
import torch.nn as nn

class ModelClass(nn.Module):
    """
    the PyTorch model to be trained by Stimulus
    """
    def __init__():
        # your model definition here
        pass

    def forward(self, mouse_dna):        
        output = model_layers(mouse_dna)

```

3. The model should include a **batch** named function that takes as input a dictionary of input "x", a dictionary of labels "y", a Callable loss function and a callable optimizer. 

In order to allow **batch** to take as input a Callable loss, we define an extra compute_loss function that parses the correct output to the correct loss class. 

```python

import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple

class ModelClass(nn.Module):
    """
    the PyTorch model to be trained by Stimulus
    """

    def __init__():
        # your model definition here
        pass

    def forward(self, mouse_dna):        
        output = model_layers(mouse_dna)

    def compute_loss_mouse_rnaseq(self, output: torch.Tensor, mouse_rnaseq: torch.Tensor, loss_fn: Callable) -> torch.Tensor:
        """
        Compute the loss.
        `output` is the output tensor of the forward pass.
        `mouse_rnaseq` is the target tensor -> label column name.
        `loss_fn` is the loss function to be used.

        IMPORTANT : the input variable "mouse_rnaseq" has the same name as the label defined in the csv above. 
        """
        return loss_fn(output, mouse_rnaseq)
    
    def batch(self, x: dict, y: dict, loss_fn: Callable, optimizer: Optional[Callable] = None) -> Tuple[torch.Tensor, dict]:
        """
        Perform one batch step.
        `x` is a dictionary with the input tensors.
        `y` is a dictionary with the target tensors.
        `loss_fn` is the loss function to be used.

        If `optimizer` is passed, it will perform the optimization step -> training step
        Otherwise, only return the forward pass output and loss -> evaluation step
        """
        output = self.forward(**x)
        loss = self.compute_loss_mouse_rnaseq(output, **y, loss_fn=loss_fn)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss, output

```

If you don't want to optimize the loss function, the code above can be written in a simplified manner

```python

import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple

class ModelClass(nn.Module):
    """
    the PyTorch model to be trained by Stimulus
    """

    def __init__():
        # your model definition here
        pass

    def forward(self, mouse_dna):        
        output = model_layers(mouse_dna)
    
    def batch(self, x: dict, y: dict, optimizer: Optional[Callable] = None) -> Tuple[torch.Tensor, dict]:
        """
        Perform one batch step.
        `x` is a dictionary with the input tensors.
        `y` is a dictionary with the target tensors.
        `loss_fn` is the loss function to be used.

        If `optimizer` is passed, it will perform the optimization step -> training step
        Otherwise, only return the forward pass output and loss -> evaluation step
        """
        output = self.forward(**x)
        loss = nn.MSELoss(output, y['mouse_rnaseq'])
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss, output

```

### Model parameter search design

### Experiment design

The file in which all information about how to handle the data before tuning is called an `experiment_config`. This file in `.json` format for now but it will be soon moved to `.yaml`. So this section could vary in the future. 

The `experiment_config` is a mandatory input for the pipeline and can be passed with the flag `--exp_conf` followed by the `PATH` of the file you want to use. Two examples of `experiment_config` can be found in the `examples` directory. 

### Experiment config content description.

