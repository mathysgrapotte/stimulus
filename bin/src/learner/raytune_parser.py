import json
import torch


class TuneParser():
    def __init__(self, results):
        """
        `results` is the output of ray.tune.
        """
        self.results = results

    def get_best_config(self):
        """
        Get the best config from the results.
        """
        return self.results.get_best_result().config
    
    def save_best_config(self, output):
        """
        Save the best config to a file.
        """
        with open(output, "w") as f:
            json.dump(self.get_best_config(), f)

    def save_best_metrics_dataframe(self, output):
        """
        Save the dataframe with the metrics at each iteration of the best sample to a file.
        """
        df = self.results.get_best_result().metrics_dataframe
        columns = [col for col in df.columns if "config" not in col]
        df = df[columns]
        df.to_csv(output, index=False)

    # def get_best_model(self):
    #     """
    #     Get the best model from the results.
    #     """
    #     return self.results.get_best_model()

    # def save_best_model(self, output):
    #     """
    #     Save the best model to a file.
    #     """
    #     with open(output, "wb") as f:
    #         f.write(self.get_best_model())