import importlib.util
import os
import src.data.experiments as exp

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