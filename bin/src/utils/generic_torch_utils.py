import torch
import numpy as np
import random


def ensure_at_least_1d(tensor: torch.Tensor) -> torch.Tensor:
    """
    Function to make sure tensors given are not zero dimensional. if they are add one dimension.
    """
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    return tensor


def set_general_seeds(seed_value : int) -> None:
    """
    Function that sets all the relevant seeds to a given value. Especially usefull in case of ray.tune.
    Ray does not have a "generic" seed as far as ray 2.23
    """

    # Set python seed
    random.seed(seed_value)
        
    # set numpy seed
    np.random.seed(seed_value)
        
    # set torch seed
    torch.manual_seed(seed_value)