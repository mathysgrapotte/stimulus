import importlib.util
import os
import src.data.experiments as exp
import math

from typing import  Union

def import_class_from_file(file_path: str) -> type:

    # Extract directory path and file name
    directory, file_name = os.path.split(file_path)
    module_name = os.path.splitext(file_name)[0]  # Remove extension to get module name
    
    # Create a module from the file path
    # In summary, these three lines of code are responsible for creating a module specification based on a file location, creating a module object from that specification, and then executing the module's code to populate the module object with the definitions from the Python file.
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find the class dynamically
    for name in dir(module):
        model_class = getattr(module, name)
        if isinstance(model_class, type) and name.startswith('Model'):
            return model_class
    
    # Class not found
    raise ImportError("No class starting with 'Model' found in the file.")


def get_experiment(experiment_name: str) -> object:
    experiment_object = getattr(exp, experiment_name)()
    return experiment_object


def memory_split_for_ray_init(memory_str:  Union[str, None]) -> float:
    """
    compute the memory requirements for ray init. 
    Usefull in case ray detects them wrongly.
    Memory is split in two for ray: for store_object memory and the other actual memory for tuning.
    The following function takes the total possible usable/allocated memory as a string parameter and returns in bytes the values for store_memory (30% as default in ray) and memory (70%).
    """

    if memory_str is None:
        return None, None

    units = {"B": 1, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40, "P": 2**50}
    
    # Extract the numerical part and the unit
    value_str = ""
    unit = ""
    
    for char in memory_str:
        if char.isdigit() or char == ".":
            value_str += char
        elif char.isalpha():
            unit += char.upper()
    
    value = float(value_str)
    
    # Normalize the unit (to handle cases like Gi, GB, Mi, etc.)
    if unit.endswith(("I", "i", "B", "b")):
        unit = unit[:-1]
    
    if unit not in units:
        raise ValueError(f"Unknown unit: {unit}")
    
    bytes_value = value * units[unit]
    
    # Calculate 30% and 70%
    thirty_percent = math.floor(bytes_value * 0.30)
    seventy_percent = math.floor(bytes_value * 0.70)
    
    return float(thirty_percent), float(seventy_percent)
