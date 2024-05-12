import math
import pandas as pd
from matplotlib import pyplot as plt
from typing import Any, Tuple
from src.learner.predict import PredictWrapper

class AnalysisPerformanceTune:
    """
    Report the performance during tuning.
    """
    def __init__(self, results_path: str):
        self.results = pd.read_csv(results_path)

    def plot_metric_vs_iteration(self, metrics: list, figsize: tuple = (10,10), output: str = None):

        # create figure
        rows, cols = self.get_grid_shape(len(metrics))
        fig, axs = plt.subplots(rows, cols, figsize=figsize)

        # plot each metric
        for i,ax in enumerate(axs.flat):
            if i >= len(metrics):
                ax.axis('off')
                continue
            self.plot_metric_vs_iteration_per_metric(axs.flat[i], metrics[i])

        # add legend
        # axs.flat[0].legend()
        handles, labels = axs[0, 0].get_legend_handles_labels()  # Get handles and labels from one subplot
        plt.legend(handles, labels, loc='upper left')  # Adjust location as needed

        # save plot
        plt.tight_layout()
        if output:
            plt.savefig(output)
        plt.show()
    
    def plot_metric_vs_iteration_per_metric(self, ax: Any, metric: str):
        """
        Plot the metric vs the iteration.
        """

        # plot training performance
        ax.plot(
            self.results.training_iteration,
            self.results['train_'+metric],
            c='blue',
            label='train'
        )

        # plot validation performance
        ax.plot(
            self.results.training_iteration,
            self.results['val_'+metric],
            c='orange',
            label='val'
        )

        # TODO set x-axis labels into integer
        # plt.xticks(range(min(self.results.training_iteration), max(self.results.training_iteration)))

        # add labels
        ax.set_xlabel('epoch')
        ax.set_ylabel(metric)

        return ax
    
    @staticmethod
    def get_grid_shape(n: int) -> Tuple[int,int]:
        """Calculates rows and columns for a rectangle layout (flexible)."""
        rows = int(math.ceil(math.sqrt(n)))  # Round up the square root for rows
        cols = int(math.ceil(n / rows))  # Calculate columns based on rows
        return rows, cols

        
class AnalysisRobustness:
    def __init__(self, metrics: list, experiment: object, batch_size: int):
        self.metrics = metrics
        self.experiment = experiment
        self.batch_size = batch_size
    
    def get_performance_table(self, names: list, model_list: dict, data_list: list) -> pd.DataFrame:
        """
        Compute the performance metrics of each model on each dataset.

        `names` is a list of names that identifies each model. 
        The corresponding dataset used to train the model will also be identified equally.

        `model_list` should have the same order as `data_list`. 
        So model_list[i] is obtained by training on data_list[i].
        """
        # check same length
        if (len(names) != len(model_list)) and (len(names) != len(data_list)):
            raise ValueError("The length of the names, model_list and data_list should be the same.")
        
        # initialize
        df = pd.DataFrame()
        model_names = []

        # for each model, get the performance table, and concat
        for i,model in enumerate(model_list):
            df = pd.concat([df, self.get_performance_table_for_one_model(names, model, data_list)])
            model_names += [names[i]] * len(data_list)
        df['model'] = model_names

        return df
    
    def get_performance_table_for_one_model(self, names: list, model: object, data_list: list) -> pd.DataFrame:
        """
        Compute the performance table of one model on each dataset.
        """
        df = pd.DataFrame()
        for data_path in data_list:  # for each data, get the performance metrics, and concat
            metric_values = PredictWrapper(model, data_path, self.experiment, split=2, batch_size=self.batch_size).compute_metrics(self.metrics)
            df = pd.concat([df, pd.DataFrame(metric_values, index=[0])])
        df['data'] = names
        return df