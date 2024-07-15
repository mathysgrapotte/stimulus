import torch
import numpy as np
import random
import glob
import os


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


def get_latest_created_dir(path : str) -> str:
    """
    Function created to retrieve latest created sirectory. It wa screated to retrieve the TuneModel subdir of ray_results.
    To ray the reusult dir can be specified but it wil create a subdir in which store the results for a given tune run.
    ray_result dir is thought to be the container of many different tune runs. the point is that is npt trivial to retrieve inside ray the name of such folder so this funcxtion is set in place.
    """

    # Get a list of all directories in the given path
    dirs = [d for d in glob.glob(os.path.join(path, '*')) if os.path.isdir(d)]
    
    # Check if there are any directories
    if not dirs:
        return None
    
    # Sort directories by creation time
    dirs.sort(key=os.path.getctime, reverse=True)
    
    # Return the latest created directory
    return dirs[0]
