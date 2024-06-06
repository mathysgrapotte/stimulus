# STIMULUS

Stochastic Testing and Input Manipulation for Unbiased Learning Systems (STIMULUS) is an end-to-end nextflow based pipeline for statistically testing training procedures of machine learning models. 

Deep learning model development in natural science is an empirical and costly process. Users must define a pre-processing pipeline, an architecture, find the best parameters for said architecture and iterate over this process.

Leveraging the power of Nextflow (polyglotism, container integration, scalable on the cloud), we propose STIMULUS, an open-source software built to automatize deep learning model development for genomics.

STIMULUS takes as input a user defined PyTorch model, a dataset, a configuration file to describe the pre-processing steps to be performed, and a range of parameters for the PyTorch model.  It then transforms the data according to all possible pre-processing steps, finds the best architecture parameters for each of the transformed datasets, performs sanity checks on the models and train a minimal deep learning version for each dataset/architecture.

Those experiments are then compiled into an intuitive report, making it easier for scientists to pick the best design choice to be sent to large scale training.

# Quick start

## Code requirements

### Data

The data is provided as a csv where the header columns are in the following format : name:type:class

*name* is user given (note that it has an impact on experiment definition).

*type* is either "input", "meta", or "label". "input" types are fed into the mode, "meta" types are registered but not transformed nor fed into the models and "label" is used as a training label. 

*class* is a supported class of data for which encoding methods have been created, please raise an issue on github or contribute a PR if a class of your interest is not implemented

column header example : 

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

In STIMULUS, user input a .py file containing a model written using pytorch (see examples in bin/tests/models)

Said model of interest should obey to minor standards as follow :

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

## Model parameter search design

## Experiment design