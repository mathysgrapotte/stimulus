import yaml
import ray.tune as tune
from copy import deepcopy
from collections.abc import Callable

class YamlRayConfigLoader():
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            self.config = self.convert_config_to_ray(self.config)

    def raytune_space_selector(self, mode: Callable, space: list) -> Callable:
        # this function applies the mode function to the space, it needs to convert the space in a right way based on the mode, for instance, if the mode is "randint", the space should be a tuple of two integers and passed as *args

        if mode.__name__ == "choice":
            return mode(space)
        
        elif mode.__name__ in ["uniform", "loguniform", "quniform", "qloguniform", "qnormal", "randint"]:
            return mode(*tuple(space))
        
        else:
            raise NotImplementedError(f"Mode {mode.__name__} not implemented yet")

    def convert_raytune(self, param: dict) -> dict:
        # get the mode function from ray.tune using getattr, return an error if it is not recognized
        try:
            mode = getattr(tune, param["mode"])
        except AttributeError:
            raise AttributeError(f"Mode {param['mode']} not recognized, check the ray.tune documentation at https://docs.ray.io/en/master/tune/api_docs/suggestion.html")

        # apply the mode function to the space
        return self.raytune_space_selector(mode, param["space"])
    
    def convert_config_to_ray(self, config: dict) -> dict:
        # the config is a dictionary of dictionaries. The main dictionary keys are either model_params, loss_params or optimizer_params. 
        # The sub-dictionary keys are the parameters of the model, loss or optimizer, those params include two values, space and mode.
        # The space is the range of values to be tested, and the mode is the type of search to be done.
        # We convert the Yaml config by calling the correct function from ray.tune matching the mode, applied on the space
        # We return the config as a dictionary of dictionaries, where the values are the converted values from the space.

        new_config = deepcopy(config)
        for key in ["model_params", "loss_params", "optimizer_params", "data_params"]:
            for sub_key in config[key]:
                new_config[key][sub_key] = self.convert_raytune(config[key][sub_key])

        return new_config
    
    def get_config_instance(self) -> dict:
        # this function take a config as input and returns an instance of said config with the values sampled from the space
        # the config is a dictionary of dictionaries. The main dictionary keys are either model_params, loss_params or optimizer_params.
        # The sub-dictionary keys are the parameters of the model, loss or optimizer, those params include two values, space and mode.

        config_instance = deepcopy(self.config)
        for key in ["model_params", "loss_params", "optimizer_params", "data_params"]:
            config_instance[key] = {}
            for sub_key in self.config[key]:
                config_instance[key][sub_key] = self.config[key][sub_key].sample()

        return config_instance 
    
    def get_config(self) -> dict:
        return self.config