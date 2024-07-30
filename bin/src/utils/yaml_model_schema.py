import ray.tune as tune
import random
import yaml
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

    def raytune_sample_from(self, mode: Callable, param: dict) -> Callable:
        """
        This function applies the tune.sample_from to a given custom sampling function.
        """

        if param["function"] == "sampint":
            return mode(lambda _: self.sampint(param["sample_space"], param["n_space"]))
        
        else:
            raise NotImplementedError(f"Function {param['function']} not implemented yet")

    def convert_raytune(self, param: dict) -> dict:
        # get the mode function from ray.tune using getattr, return an error if it is not recognized
        try:
            mode = getattr(tune, param["mode"])
        except AttributeError:
            raise AttributeError(f"Mode {param['mode']} not recognized, check the ray.tune documentation at https://docs.ray.io/en/master/tune/api_docs/suggestion.html")

        # apply the mode function
        if param["mode"] != "sample_from":
            return self.raytune_space_selector(mode, param['space'])
        else:
            return self.raytune_sample_from(mode, param)
    
    def convert_config_to_ray(self, config: dict) -> dict:
        # the config is a dictionary of dictionaries. The main dictionary keys are either model_params, loss_params or optimizer_params. 
        # The sub-dictionary keys are the parameters of the model, loss or optimizer, those params include two values, space and mode.
        # The space is the range of values to be tested, and the mode is the type of search to be done.
        # We convert the Yaml config by calling the correct function from ray.tune matching the mode, applied on the space
        # We return the config as a dictionary of dictionaries, where the values are the converted values from the space.
        print(config)
        new_config = deepcopy(config)
        for key in ["model_params", "loss_params", "optimizer_params", "data_params"]:
            for sub_key in config[key]:

                # if mode is provided, it understands that it is a ray.tune parameter
                # therefore, it converts the space provided in the config to a ray.tune parameter space
                # otherwise, it keeps the value as it is. In this way, we can use the same config for both ray.tune and non-ray.tune elements (for example provide a single fixed value).
                if 'mode' in config[key][sub_key]:
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

    @staticmethod
    def sampint(sample_space: list, n_space: list) -> list:
        """
        This function returns a list of n samples from the sample_space.

        This function is specially useful when we want different number of layers,
        and each layer with different number of neurons.

        `sample_space` is the range of (int) values from which to sample
        `n_space` is the range of (int) number of samples to take
        """
        sample_space = range(sample_space[0], sample_space[1]+1)
        n_space = range(n_space[0], n_space[1]+1)
        n = random.choice(n_space)
        return random.sample(sample_space, n)
