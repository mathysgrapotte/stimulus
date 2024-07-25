import torch
import numpy as np
import random
import glob
import os

from typing import Union

def ensure_at_least_1d(tensor: torch.Tensor) -> torch.Tensor:
    """
    Function to make sure tensors given are not zero dimensional. if they are add one dimension.
    """
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    return tensor


def set_general_seeds(seed_value : Union[int, None]) -> None:
    """
    Function that sets all the relevant seeds to a given value. Especially usefull in case of ray.tune.
    Ray does not have a "generic" seed as far as ray 2.23
    """

    # Set python seed
    random.seed(seed_value)
        
    # set numpy seed
    np.random.seed(seed_value)
        
    # set torch seed, diffrently from the two above torch can nopt take Noneas input value so it will not be called in that case.
    if seed_value is not None:
        torch.manual_seed(seed_value)

