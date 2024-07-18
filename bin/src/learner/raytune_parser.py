import json
import os
import torch

class TuneParser():
    def __init__(self, results):
        """
        `results` is the output of ray.tune.
        """
        self.results = results

    def get_best_config(self) -> dict:
        """
        Get the best config from the results.
        """
        config = self.results.get_best_result().config
        return config
    
    def save_best_config(self, output: str) -> None:
        """
        Save the best config to a file.

        TODO maybe only save the relevant config values.
        """
        config = self.get_best_config()
        config = self.fix_config_values(config)
        with open(output, "w") as f:
            json.dump(config, f, indent=4)

    def fix_config_values(self, config):
        """
        Correct config values.
        """
        # fix the model and experiment values to avoid problems with serialization
        # TODO this is a quick fix to avoid the problem with serializing class objects. maybe there is a better way.
        config['model'] = config['model'].__name__       
        config['experiment'] = config['experiment'].__class__.__name__ 
        if 'tune' in config and 'tune_params' in config['tune']:
            del config['tune']['tune_params']['scheduler']
        # delete miscellaneus keys, used only during debug mode for example
        del config['_debug'], config['tune_run_path']

        return config

    def save_best_metrics_dataframe(self, output: str) -> None:
        """
        Save the dataframe with the metrics at each iteration of the best sample to a file.
        """
        df = self.results.get_best_result().metrics_dataframe
        columns = [col for col in df.columns if "config" not in col]
        df = df[columns]
        df.to_csv(output, index=False)

    def get_best_model(self) -> dict:
        """
        Get the best model weights from the results.
        """
        checkpoint = self.results.get_best_result().checkpoint.to_directory()
        checkpoint = os.path.join(checkpoint, "model.pt")
        return torch.load(checkpoint)
    
    def save_best_model(self, output: str) -> None:
        """
        Save the best model weights to a file.
        """
        torch.save(self.get_best_model(), output)

    def get_best_optimizer(self) -> dict:
        """
        Get the best optimizer state from the results.
        """
        checkpoint = self.results.get_best_result().checkpoint.to_directory()
        checkpoint = os.path.join(checkpoint, "optimizer.pt")
        return torch.load(checkpoint)

    def save_best_optimizer(self, output: str) -> None:
        """
        Save the best optimizer state to a file.
        """
        torch.save(self.get_best_optimizer(), output)
        