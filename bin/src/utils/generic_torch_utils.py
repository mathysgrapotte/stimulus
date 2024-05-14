import torch

def ensure_at_least_1d(tensor: torch.Tensor) -> torch.Tensor:
    """
    Function to make sure tensors given are not zero dimensional. if they are add one dimension.
    """
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    return tensor